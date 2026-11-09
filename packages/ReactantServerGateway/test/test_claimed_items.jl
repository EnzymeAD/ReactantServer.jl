# A request's batch extent is the client's claim (GW-4). The gateway never rejects a request over it
# (the worker is the authority on validity), but it counts no more items than the request carries
# bytes of data for the batch input, and the work counters it charges cannot throw or drift whatever
# the count. Before, a claim of 2^62 rows overflowed the arithmetic (`InexactError`, so concurrent
# requests failed with INTERNAL) and the Float64 counters lost the other charges in flight beside it,
# leaving a worker below zero and favored for good. Reuses `_body` and `_poll` from
# test_scheduler.jl and `_pk_state` from test_lpt_packing.jl.

import ProtoBuf as CPB

const _CInf = ReactantServerGateway.inference

_cenc(msg) = (io = IOBuffer(); CPB.encode(CPB.ProtoEncoder(io), msg); take!(io))
_cparam(v::Int64) = _CInf.InferParameter(; parameter_choice = CPB.OneOf(:int64_param, v))
_cparam(v::String) = _CInf.InferParameter(; parameter_choice = CPB.OneOf(:string_param, v))

# A request whose batch input lives in shared memory, declaring `bytes` there.
function _shm_body(shape, bytes)
    t = _CInf.var"ModelInferRequest.InferInputTensor"(;
        name = "IN", datatype = "UINT8", shape = Int64[shape...],
        parameters = Dict(
            "shared_memory_region" => _cparam("client-slot-0"),
            "shared_memory_byte_size" => _cparam(Int64(bytes)),
        )
    )
    return _cenc(_CInf.ModelInferRequest(; model_name = "m", inputs = [t]))
end

@testset "request_item_count: the claimed extent, capped at the data the request carries" begin
    count(body) = GW.request_item_count(body, "IN", 1)
    for claim in (2^50, 2^62)
        body = _body("m", "IN", (claim, 4); bytes = 64)
        @test GW.peek_batch_size(body, "IN", 1) == claim    # the claim is read as sent...
        @test count(body) == 64                              # ...and believed only as far as the bytes go
        @test GW.request_item_count(body, "IN", 2) == 4     # a smaller extent stands
    end
    # A real request is counted as it says.
    @test count(_body("m", "IN", (32, 4))) == 32

    # Inline typed contents count by their encoded length: no row takes less than a byte.
    contents = _CInf.InferTensorContents(; fp32_contents = Float32[1:8;])
    typed(shape) = _cenc(
        _CInf.ModelInferRequest(;
            model_name = "m",
            inputs = [_CInf.var"ModelInferRequest.InferInputTensor"(; name = "IN", datatype = "FP32", shape = Int64[shape...], contents = contents)]
        )
    )
    @test count(typed((2^62, 1))) == length(_cenc(contents))
    @test count(typed((8, 1))) == 8

    # raw_input_contents is positional, so the batch input's own entry is the one that counts.
    two = _cenc(
        _CInf.ModelInferRequest(;
            model_name = "m",
            inputs = [
                _CInf.var"ModelInferRequest.InferInputTensor"(; name = "OTHER", datatype = "UINT8", shape = Int64[4096]),
                _CInf.var"ModelInferRequest.InferInputTensor"(; name = "IN", datatype = "UINT8", shape = Int64[2^62]),
            ],
            raw_input_contents = [zeros(UInt8, 4096), zeros(UInt8, 3)]
        )
    )
    @test count(two) == 3

    # Shared memory: the byte size the request declares, and never past the ceiling.
    @test count(_shm_body((2^62,), 1024)) == 1024
    @test count(_shm_body((2^62,), 2^62)) == GW.MAX_REQUEST_ITEMS
    @test count(_shm_body((16,), 2^62)) == 16

    # What cannot be believed counts as one item: no data, a negative or zero extent, no such input,
    # an unreadable body. None of it is rejected; the worker decides.
    @test count(_body("m", "IN", (2^62,); bytes = 0)) == 1
    @test count(_body("m", "IN", (-5, 4); bytes = 64)) == 1
    @test count(_body("m", "OTHER", (8,))) == 1
    @test count(_body("m", "IN", (2^62,); bytes = 64)[1:8]) == 1     # cut inside the input tensor
end

# Drive `select`/`release` from many threads at once, mixing requests that claim 2^50 and 2^62 rows
# with ordinary ones, and return the errors the tasks hit.
function _hammer(select, release, bodies; tasks = 8, rounds = 200)
    errors = Any[]
    lk = ReentrantLock()
    @sync for t in 1:tasks
        Threads.@spawn try
            for r in 1:rounds
                held = select(bodies[mod1(t + r, length(bodies))])
                yield()
                release(held)
            end
        catch e
            @lock lk push!(errors, e)
        end
    end
    return errors
end

const _CLAIM_BODIES = [
    _body("m", "IN", (2^50, 4); bytes = 64),
    _body("m", "IN", (4, 4)),
    _body("m", "IN", (2^62, 4); bytes = 4096),
    _shm_body((2^62,), 2^62),                 # the one case whose count reaches the ceiling
    _body("m", "IN", (1, 4)),
]

@testset "least_outstanding: claimed extents of 2^50 and 2^62 neither throw nor drift ($basis)" for basis in (:items, :compute)
    rm = GW.RoutingMeta(30.0)
    GW.refresh_routing_meta!(rm, _poll(Dict("m" => (0.5, 1, 1)); batch_at = Dict("m" => ("IN", 1))))
    s = GW.LeastOutstandingScheduler(rm; basis = basis)
    cfg = _sched_cfg("least_outstanding")
    pool = GW.ClientPool(cfg)
    routes = GW.DiscoveredRoutes()
    GW.swap_table!(routes, GW.RoutingTable(Dict("m" => ["127.0.0.1:7001", "127.0.0.1:7002"])))
    metrics = GW.GatewayMetrics()
    refresher = GW.RouteRefresher(pool, routes, metrics)
    try
        @test GW.request_units(s, "m", _CLAIM_BODIES[1]) == 64
        select(body) = GW.select_replicas(
            s, GW.ScheduleContext("m", "id", GW.request_units(s, "m", body), pool, routes, metrics, refresher)
        )[2]
        errors = _hammer(select, r -> GW.release!(s, r), _CLAIM_BODIES)
        @test isempty(errors)
        @test all(c -> c[] == 0, values(@atomic s.inflight))
        @test length(@atomic s.inflight) == 2
    finally
        GW.close_pool!(pool)
    end
end

@testset "lpt_packing: claimed extents of 2^50 and 2^62 neither throw nor drift ($basis)" for basis in ("items", "compute")
    s = _pk_state(;
        routing_policy = "fill_least", work_basis = basis, max_batch = 8, batch_at = ("IN", 1),
        item_costs = Dict("m" => 0.5)
    )
    @test GW.request_units(s, "m", _CLAIM_BODIES[1]) == 64
    @test GW.request_units(s, "m", _CLAIM_BODIES[3]) == 4096
    select(body) = GW.route_replica(s, "m", GW.request_units(s, "m", body))[2]
    errors = _hammer(select, GW._release_route!, _CLAIM_BODIES)
    @test isempty(errors)
    @test all(a -> a[] == 0, values(@atomic s.worker_load))
    @test all(a -> a[] == 0, values(@atomic s.outstanding))
end
