# Precompile workload: run the worker's request path once at package build time, against the mock
# backend, so the compiled code lands in the package image and a fresh worker does not compile it on
# its first requests. Measured on a CPU worker before this existed: the first ModelInfer of a process
# compiled for about 4 s (protobuf decode, the handler, response encode), and the first metrics scrape
# and DescribeModels call for about 5 s more.
#
# What it covers is package code specialized on concrete types: the raw ModelInfer handler with u8,
# f32 and i64 tensors at ranks 2 and 4 (decode, scheduling, batching and padding, the mock execution,
# encode), one typed-contents request, the control and metadata handlers with their gRPC
# serialization, and a full metrics exposition. What it cannot cover is code that only exists at run
# time: a bundle's model.jl, and the HTTP/gRPC server closures specialized on a live server. It opens
# no sockets and leaves no task running.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # One mock model: `inputs` and `outputs` as (name, dtype, Julia shape with -1 for the batch axis),
    # executed by `fn` against a single f32 weight vector of length 2, compiled at batch sizes 1 and 4.
    function _pc_entry(name, inputs, outputs, fn)
        spec(n, dt, shape) = TensorSpec(
            n, dt, Dim[s < 0 ? Dim(BATCH) : Dim(FIXED, s) for s in shape], findfirst(<(0), shape)
        )
        ins = TensorSpec[spec(i...) for i in inputs]
        outs = TensorSpec[spec(o...) for o in outputs]
        bdim = findfirst(<(0), inputs[1][3]) - 1
        man = Manifest(
            "2.0", name, "", ins, outs, nothing, nothing, BatchingSpec(Int[]),
            Provenance(Dict{String, Any}()), bdim
        )
        sig = ModelSignature(
            String[i[1] for i in inputs], DataType[julia_type(i[2]) for i in inputs], ["w"],
            length(outputs), String[o[1] for o in outputs], DataType[julia_type(o[2]) for o in outputs], bdim
        )
        exec = MockExecutable(fn, length(outputs))
        model = LoadedModel(sig, Dict{Int, Any}(1 => exec, 4 => exec), Any[MockBuffer(Float32[2, 3])])
        entry = ModelEntry(name, man, Dict{Int, Vector{UInt8}}(), "", nothing, model, nothing, identity, identity)
        entry.sched = ModelSchedState(name, ModelSchedConfig(1.0), 0.0)
        return entry
    end
    # Row-major wire tensor and its raw bytes for a Julia array.
    _pc_input(name, a::Array) = (
        inference.var"ModelInferRequest.InferInputTensor"(;
            name = name, datatype = kserve_string(dtype_of(eltype(a))),
            shape = Int64[reverse(size(a))...]
        ),
        collect(reinterpret(UInt8, vec(a))),
    )
    function _pc_request(model, named, outputs = String[])
        pairs = [_pc_input(n, a) for (n, a) in named]
        req = inference.ModelInferRequest(;
            model_name = model, inputs = [p[1] for p in pairs],
            outputs = [inference.var"ModelInferRequest.InferRequestedOutputTensor"(; name = o) for o in outputs],
            raw_input_contents = [p[2] for p in pairs]
        )
        return _encode_message(req)
    end

    # The live servers: the gRPC server and the metrics listener on loopback ports the OS assigns,
    # each answering real requests, then stopped. HTTP.jl compiles its listener and stream-dispatch
    # closures specialized on these exact server and handler types (about 5 s on a worker's first
    # connections), which only a running server reaches. The gRPC calls go over HTTP.jl's own HTTP/2
    # client, so the workload needs no gRPC client package.
    _pc_grpc_frame(msg::Vector{UInt8}) =
        vcat(UInt8[0], reinterpret(UInt8, [hton(UInt32(length(msg)))]), msg)
    function _pc_serve(sched, reg, ctx, metrics)
        server = build_grpc_server(sched, reg, "mock", ctx.shm, "127.0.0.1", 1, ctx)
        server.port = 0                       # an OS-assigned port, read back after start!
        gRPCServer.add_interceptor!(server, _DrainInterceptor(ReactantServerCore.DrainGate()))   # as serve does
        mserver = nothing
        client = HTTP.Client()
        gRPCServer.start!(server)
        try
            port = something(gRPCServer.bound_port(server), 0)
            t_end = monotonic_time() + 10
            while port == 0 && monotonic_time() < t_end   # the bound port appears once start! has bound
                sleep(0.01)
                port = something(gRPCServer.bound_port(server), 0)
            end
            headers = ["content-type" => "application/grpc", "te" => "trailers"]
            for (path, msg) in (
                    ("inference.GRPCInferenceService/ServerReady", UInt8[]),
                    ("inference.GRPCInferenceService/ModelInfer", _pc_request("pc_f32", ["x" => rand(Float32, 2, 1)])),
                )
                HTTP.request(
                    "POST", "http://127.0.0.1:$port/$path"; client = client, protocol = :h2,
                    headers = headers, body = _pc_grpc_frame(msg)
                )
            end
            mserver = start_worker_metrics(metrics, "127.0.0.1", 0; ready_fn = _RegistryReady(reg, sched), live_fn = _DispatchLive(sched), worker_name = "worker0", gpu = "0")
            mport = HTTP.port(mserver)
            for path in ("/metrics", "/readyz")
                HTTP.request("GET", "http://127.0.0.1:$mport$path"; client = client, protocol = :h1)
            end
        finally
            HTTP.close_idle_connections!(client)
            mserver === nothing || close(mserver)
            gRPCServer.stop!(server; force = true)
        end
        return nothing
    end

    # The model-directory watcher's poll loop is a closure inside start_watching!, so it compiles when
    # a watcher first runs (about 0.8 s at startup). Run one over an empty directory for a few polls.
    function _pc_watch(sched, backend, pool)
        mktempdir() do dir
            cfg = ServerConfig(
                [dir], "", RuntimeConfig(CPU_BACKEND, 0, 0.9, true, true), sched.cfg,
                EndpointsConfig("127.0.0.1", 0)
            )
            w = BundleWatcher(sched, backend, pool, cfg; interval = 0.05, on_demand = false)
            start_watching!(w)
            sleep(0.3)
            stop_watching!(w)
            wait(w.task)
        end
        return nothing
    end

    @compile_workload begin
        backend = MockBackend()
        pool = MemoryPool(backend, MockClient(), MockDevice(0), "mock", nothing)
        reg = ModelRegistry()
        for e in (
                # (features, n) f32 -> f32
                _pc_entry("pc_f32", [("x", F32, [2, -1])], [("y", F32, [2, -1])], a -> [a[1] .* a[2]]),
                # an image-shaped (w, h, c, n) u8 input -> (2, n) f32 scores
                _pc_entry(
                    "pc_u8", [("x", U8, [4, 4, 1, -1])], [("y", F32, [2, -1])],
                    a -> [a[2] .* reshape(Float32.(vec(sum(a[1]; dims = (1, 2, 3)))), 1, :)]
                ),
                # (features, n) f32 -> (1, n) i64, plus a second f32 output
                _pc_entry(
                    "pc_i64", [("x", F32, [2, -1])], [("k", I64, [1, -1]), ("y", F32, [2, -1])],
                    a -> [round.(Int64, sum(a[1]; dims = 1)), a[1] .* a[2]]
                ),
            )
            reg.by_name[e.name] = e
        end
        sched = Scheduler(reg, backend, pool, SchedulerConfig(30.0, 64, 30.0))
        cfg = ServerConfig(
            ["."], "", RuntimeConfig(CPU_BACKEND, 0, 0.9, true, true), sched.cfg,
            EndpointsConfig("127.0.0.1", 0)
        )
        metrics = WorkerMetrics(sched, backend, pool, cfg; worker_name = "worker0")
        ctx = InferContext(sched, reg, SharedMemoryRegistry(), "mock", metrics)
        start!(sched)
        try
            # One row and a short batch padded up to the compiled size 4, with and without named outputs.
            for n in (1, 3)
                _handle_infer_raw(ctx, _pc_request("pc_f32", ["x" => rand(Float32, 2, n)]), 0)
                _handle_infer_raw(ctx, _pc_request("pc_u8", ["x" => rand(UInt8, 4, 4, 1, n)], ["y"]), 0)
                _handle_infer_raw(ctx, _pc_request("pc_i64", ["x" => rand(Float32, 2, n)], ["k", "y"]), 0)
            end
            # Typed contents, as a client that does not send raw bytes does.
            typed = inference.ModelInferRequest(;
                model_name = "pc_f32",
                inputs = [
                    inference.var"ModelInferRequest.InferInputTensor"(;
                        name = "x", datatype = "FP32", shape = Int64[1, 2],
                        contents = inference.InferTensorContents(; fp32_contents = Float32[1, 2])
                    ),
                ]
            )
            _handle_infer_raw(ctx, _encode_message(typed), 0)
            # The control and metadata answers, serialized the way gRPCServer does for typed handlers.
            for resp in (
                    _handle_describe_models(ctx, control.DescribeModelsRequest()),
                    _handle_model_control_status(ctx),
                    _handle_repository_index(ctx, inference.RepositoryIndexRequest()),
                    _handle_model_metadata(ctx, inference.ModelMetadataRequest(; name = "pc_u8")),
                    _handle_server_ready(ctx),
                )
                gRPCServer.serialize_message(resp; raw = false)
            end
            io = IOBuffer()
            Prometheus.expose(io, metrics.registry)
            inject_metric_labels(String(take!(io)), ["worker" => "worker0", "gpu" => "0"])
            _pc_serve(sched, reg, ctx, metrics)
            _pc_watch(sched, backend, pool)
        finally
            shutdown!(sched)
        end
    end
end

# Directives for the startup path, which runs only against a real backend (a Reactant client, model
# compilation, device weights) and so cannot be driven by the mock workload above. `precompile`
# infers and compiles a method and what it calls without running it. Signatures are the ones a
# worker compiled at startup in a --trace-compile run, plus the shared host-weight store and the
# on-demand weight cache, which a GPU node uses and a CPU run does not. A signature that stops
# matching (a renamed keyword, say) only costs the compile time back, so it is reported, not fatal.
let RS = ReactantServerCore, ok = Bool[]
    kw(names, types) = NamedTuple{names, Tuple{types...}}
    push!(ok, precompile(Tuple{typeof(serve), String}))
    push!(
        ok, precompile(
            Tuple{
                typeof(Core.kwcall), kw((:backend, :blocking, :worker_name), (ReactantBackend, Bool, String)),
                typeof(serve), ServerConfig,
            }
        )
    )
    push!(
        ok, precompile(
            Tuple{
                typeof(Core.kwcall),
                kw((:device_memory_fraction, :preallocate, :autotune_cache, :autotune_cache_dir), (Float64, Bool, Nothing, String)),
                typeof(make_client), ReactantBackend, String,
            }
        )
    )
    for store in (RS.PrivateWeightStore, RS.SharedWeightStore)
        push!(
            ok, precompile(
                Tuple{
                    typeof(Core.kwcall),
                    kw((:state, :on_demand, :store, :executable_cache), (RS.ResidencyState, Bool, store, Bool)),
                    typeof(build_loaded_model), ReactantBackend, MemoryPool, ModelEntry,
                }
            )
        )
    end
    for cache in (Nothing, WeightCache)
        push!(
            ok, precompile(
                Tuple{
                    typeof(Core.kwcall), kw((:registry, :weight_cache), (ModelRegistry, cache)),
                    typeof(memory_report), ReactantBackend, MemoryPool,
                }
            )
        )
    end
    push!(ok, precompile(Tuple{typeof(run_model), ReactantBackend, MemoryPool, LoadedModel, Vector{NamedTensor}}))
    # The "models ready" log line: the logging path specializes on its keyword NamedTuple.
    push!(ok, precompile(Tuple{typeof(pairs), kw((:count, :control_mode, :memory), (Int, RS.ModelControlMode, String))}))
    push!(ok, precompile(Tuple{typeof(weight_order), WeightsFile}))
    push!(ok, precompile(Tuple{typeof(weights_nbytes), WeightsFile, Vector{String}}))
    push!(ok, precompile(Tuple{typeof(load_pinned_weights), ReactantBackend, MemoryPool, WeightsFile, Vector{String}}))
    all(ok) || @warn "ReactantServer precompile: some startup directives no longer match a method" failed = findall(!, ok)
end
