# merge_expositions reads byte ranges where it used to make a String of every line, and must write
# exactly what it wrote before for every input. The implementation it replaced is kept below as the
# oracle, and the two are compared on production expositions, on the cases the merge has to get
# right, and on seeded random expositions.
#
# data/worker0.prom and data/worker1.prom are two production worker expositions (about 2,100
# series each), trimmed: every julia_* and process_* family (the gateway exports them too), the
# worker families worker_dispatch_total, worker_info, worker_interactive_lag_max_seconds,
# worker_interactive_lag_seconds, worker_models_loaded, worker_queue_wait_seconds,
# worker_request_latency_seconds, worker_requests_total and worker_weight_stalled_reads (whose
# headerless worker_weight_stalled_reads_total follows it), with model-labelled samples kept for
# three models only, renamed model_001, model_002 and model_003 in their sort order.
# Lines are otherwise verbatim and in their order.

# merge_expositions as of 9461ddee, verbatim but for its name.
function _merge_expositions_before(texts::Vector{String})
    order = String[]                                  # family emission order
    help = Dict{String, String}()                      # family -> "# HELP ..." line
    type = Dict{String, String}()                      # family -> "# TYPE ..." line
    samples = Dict{String, Vector{String}}()           # family -> sample lines
    current = ""
    family_of(line) = begin
        # Sample lines belong to the family of the preceding header when their name extends it
        # (histogram _bucket/_sum/_count); otherwise the bare metric name is its own family.
        stop = something(findfirst(c -> c == '{' || c == ' ', line), lastindex(line) + 1)
        name = line[1:(stop - 1)]
        (!isempty(current) && (name == current || startswith(name, current * "_"))) ? current : name
    end
    ensure!(fam) = fam in order || push!(order, fam)
    for text in texts
        current = ""
        for line in eachline(IOBuffer(text))
            isempty(strip(line)) && continue
            if startswith(line, "# HELP ")
                current = split(line; limit = 4)[3]
                ensure!(current)
                get!(help, current, line)
            elseif startswith(line, "# TYPE ")
                current = split(line; limit = 4)[3]
                ensure!(current)
                get!(type, current, line)
            elseif startswith(line, '#')
                continue
            else
                fam = family_of(line)
                ensure!(fam)
                push!(get!(samples, fam, String[]), line)
            end
        end
    end
    io = IOBuffer()
    for fam in order
        haskey(help, fam) && println(io, help[fam])
        haskey(type, fam) && println(io, type[fam])
        for s in get(samples, fam, String[])
            println(io, s)
        end
    end
    return String(take!(io))
end

# What a merge gives: its output, or the type of what it threw. Some malformed lines (a header with
# no name, a metric name ending in a non-ASCII character) throw, and must go on throwing the same.
function _merge_outcome(merge, texts)
    try
        return merge(texts)
    catch e
        return typeof(e)
    end
end

# Whether the merge matches the oracle on `texts`, given as Strings and as the byte vectors the
# scrape path passes.
function _merges_as_before(texts::Vector{String})
    want = _merge_outcome(_merge_expositions_before, texts)
    bytes = Vector{UInt8}[Vector{UInt8}(t) for t in texts]
    return _merge_outcome(ReactantServerGateway.merge_expositions, texts) == want &&
        _merge_outcome(ReactantServerGateway.merge_expositions, bytes) == want
end

# A seeded xorshift64 generator, so the random cases are the same on every run without the Random
# package: `_pick!(g, xs)` is a pseudo-random element of `xs`.
mutable struct _Xorshift
    s::UInt64
end
function _pick!(g::_Xorshift, xs)
    x = g.s
    x ⊻= x << 13
    x ⊻= x >> 7
    x ⊻= x << 17
    g.s = x
    return xs[Int(x % UInt64(length(xs))) + 1]
end

@testset "merge_expositions writes what it always has" begin
    @testset "production expositions" begin
        w0 = read(joinpath(@__DIR__, "data", "worker0.prom"), String)
        w1 = read(joinpath(@__DIR__, "data", "worker1.prom"), String)
        # The gateway's own exposition, with its runtime families and a few of its series.
        m = ReactantServerGateway.GatewayMetrics(Dict("127.0.0.1:9100" => "worker0", "127.0.0.1:9101" => "worker1"))
        ReactantServerGateway.register_process_metrics!(m)
        for (ep, model) in ("127.0.0.1:9100" => "model_002", "127.0.0.1:9101" => "model_001")
            ReactantServerGateway.inc_requests!(m, "ModelInfer", model, "OK")
            ReactantServerGateway.observe_request!(m, "ModelInfer", model, 0.02)
            ReactantServerGateway.observe_worker!(m, "ModelInfer", ep, 0.015)
            ReactantServerGateway.set_worker_ready!(m, ep, true)
            ReactantServerGateway.set_placement_weight!(m, model, ep, 1.0)
        end
        io = IOBuffer()
        ReactantServerGateway.expose(io, m)
        gw = String(take!(io))
        for texts in ([gw, w0, w1], [w0, w1], [w1, w0, gw], [gw], [w0], [w0, w0], [gw, gw])
            @test _merges_as_before(texts)
        end
        merged = ReactantServerGateway.merge_expositions([gw, w0, w1])
        # A family the gateway and the workers all export keeps the gateway's header, with every
        # source's samples under it.
        @test count("# TYPE julia_gc_live_bytes gauge", merged) == 1
        @test occursin(r"# TYPE julia_gc_live_bytes gauge\njulia_gc_live_bytes \d+\njulia_gc_live_bytes\{gpu=\"0\",worker=\"worker0\"\} \d+\njulia_gc_live_bytes\{gpu=\"1\",worker=\"worker1\"\} \d+\n", merged)
        # The workers' headerless counter stays with the gauge whose name it extends.
        @test occursin(
            """
            worker_weight_stalled_reads{gpu="0",worker="worker0"} 0
            worker_weight_stalled_reads_total{gpu="0",worker="worker0"} 0
            worker_weight_stalled_reads{gpu="1",worker="worker1"} 0
            worker_weight_stalled_reads_total{gpu="1",worker="worker1"} 0
            """, merged
        )
    end

    @testset "edge cases" begin
        cases = [
            # A family in only some inputs, and inputs that are empty or hold nothing but blanks.
            String[],
            ["", "", ""],
            ["\n\n", " \t\n", "\u2003\n"],
            ["# HELP a A.\n# TYPE a counter\na 1\n", "# HELP b B.\nb 1\n", "# TYPE a counter\na 2\n"],
            # TYPE before HELP, and a later source's different HELP and TYPE (the first wins).
            ["# TYPE a gauge\n# HELP a A.\na 1\n", "# HELP a Another.\n# TYPE a counter\na 2\n"],
            # A family seen as a bare sample before any source declares it.
            ["a 1\n", "# HELP a A.\n# TYPE a counter\na 2\n"],
            # Histogram and summary suffixes, in one source and across two.
            [
                "# HELP h H.\n# TYPE h histogram\nh_bucket{le=\"1\"} 1\nh_bucket{le=\"+Inf\"} 2\nh_sum 3\nh_count 2\n",
                "# HELP s S.\n# TYPE s summary\ns{quantile=\"0.5\"} 1\ns_sum 2\ns_count 3\n" *
                    "# HELP h H.\n# TYPE h histogram\nh_bucket{le=\"+Inf\"} 4\nh_sum 5\nh_count 4\n",
            ],
            # Family names that are prefixes of others: a_b under a, but ab, a_ and a_b_c by name
            # once their own header or another family's comes between.
            ["# HELP a A.\na 1\na_b 2\n# HELP a_b AB.\na_b 3\na_bc 4\n# HELP ab X.\nab 5\na_b_c 6\na_ 7\n"],
            ["# HELP a A.\n# TYPE a gauge\na 1\n", "# HELP a_b AB.\na_b 2\na 3\n"],
            # Comments, blank lines, a header with extra whitespace, and one that is only a comment.
            ["\n# a comment\n\n# HELP  a \t A.\n\n# EOF\n#HELP b x\na 1\n# HELP\n\na 2\n"],
            # A missing trailing newline, CRLF line ends, and a '\r' that is not a line end.
            ["# HELP a A.\na 1", "a 2\r\na 3\r\n", "a 4\r", "a\r5\n"],
            # Label quoting and escaping, and a value with a space.
            ["# HELP a A \\\\ \\n.\na{x=\"a b\",y=\"c\\\"d\",z=\"e\\\\\\nf\"} 1\na{x=\"{ }\"} 2 1700000000\n"],
            # Unicode in HELP text, label values and names, and Unicode whitespace.
            ["# HELP a Ä 😀.\na{m=\"é\"} 1\naé 2\n# HELP b\u00a0B.\nb 3\n\u00a0\u2003\n"],
            # Malformed lines: the merge throws, as it always has.
            ["# HELP \n"],
            ["# TYPE \t\na 1\n"],
            ["é 1\n"],
            ["\xe0\x82\xa0 1\n"],
        ]
        for texts in cases
            @test _merges_as_before(texts)
        end
        # A few outputs spelled out.
        @test ReactantServerGateway.merge_expositions(cases[4]) ==
            "# HELP a A.\n# TYPE a counter\na 1\na 2\n# HELP b B.\nb 1\n"
        @test ReactantServerGateway.merge_expositions(cases[5]) == "# HELP a A.\n# TYPE a gauge\na 1\na 2\n"
        @test ReactantServerGateway.merge_expositions(cases[11]) == "# HELP a A.\na 1\na 2\na 3\na 4\r\na\r5\n"
        @test ReactantServerGateway.merge_expositions(cases[8]) ==
            "# HELP a A.\na 1\na_b 2\n# HELP a_b AB.\na_b 3\na_bc 4\n# HELP ab X.\nab 5\na_b_c 6\na_ 7\n"
        @test_throws BoundsError ReactantServerGateway.merge_expositions(cases[14])
    end

    @testset "random expositions" begin
        lines = [
            "# HELP a A.", "# TYPE a counter", "# HELP a Another.", "# TYPE a gauge", "a 1", "a{x=\"y\"} 2",
            "# HELP a_b AB, ü.", "# TYPE a_b gauge", "a_b 3", "a_b_total 4", "ab 5", "a_ 6",
            "# HELP h H.", "# TYPE h histogram", "h_bucket{le=\"0.1\"} 1", "h_bucket{le=\"+Inf\"} 2",
            "h_sum 0.3", "h_count 2", "hx 1", "h_x 2",
            "# HELP s S.", "# TYPE s summary", "s{quantile=\"0.5\"} 1", "s_sum 2", "s_count 3",
            "# HELP  a  Spaced.", "# TYPE\ta counter", "# HELP a\u00a0Unicode space.", "# a comment", "#", "# EOF",
            "", " ", "\t", "\u2003", "a{x=\"p q\",y=\"r\\\"s\"} 5", "a{x=\"\\\\\\n\"} 6", "a{x=\"é ü\"} 7",
            "aéb 8", " a 9", "{x=\"y\"} 10", "a\r11", "a\x00 12",
        ]
        malformed = ["# HELP ", "# TYPE \t", "é 1", "fé{x=\"y\"} 1", "\xc0\xa0 1", "\xe0\x82\xa0 1"]
        ends = ["\n", "\n", "\n", "\n", "\r\n", "\n\n"]
        g = _Xorshift(0x5eed_2026_1111_0001)
        line!() = _pick!(g, 1:100) == 1 ? _pick!(g, malformed) : _pick!(g, lines)
        text!() = join([line!() * _pick!(g, ends) for _ in 1:_pick!(g, 0:12)]) * _pick!(g, ["", "", "a 0"])
        cases = [String[text!() for _ in 1:_pick!(g, 0:4)] for _ in 1:3000]
        mismatched = filter(!_merges_as_before, cases)
        @test isempty(mismatched)
        isempty(mismatched) || @info "merge_expositions differs from the oracle" first(mismatched, 3)
        # Most cases merge; the malformed lines make some throw, and those are compared too.
        @test count(t -> _merge_outcome(_merge_expositions_before, t) isa String, cases) > 2000
    end
end
