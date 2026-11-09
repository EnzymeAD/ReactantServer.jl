# Clients choose model names and shared-memory region names, so neither may become a metric label
# value as sent (GW-5): every name a client sent used to add a counter and an 18-bucket histogram,
# and a region name, random per client process, labelled the shared-memory series. A model is
# labelled by its name only while the routing table holds it; every other name is counted under
# `_unknown`, and the shared-memory RPCs carry no model label. The lpt_packing arrival counters,
# one per name, are kept only for routed models. Reuses `_sched_cfg` from test_scheduler.jl and
# `_pk_state` from test_lpt_packing.jl.

import ProtoBuf as LPB

const _LInf = ReactantServerGateway.inference

_lenc(msg) = (io = IOBuffer(); LPB.encode(LPB.ProtoEncoder(io), msg); take!(io))
_lrandname() = string(rand(UInt128); base = 36)

# The series of `family` in an exposition of `metrics`, as their label sets.
function _series_lines(metrics, family)
    io = IOBuffer()
    GW.expose(io, metrics)
    return Set(l[(length(family) + 2):(findfirst('}', l) - 1)] for l in split(String(take!(io)), '\n') if startswith(l, family * "{"))
end

@testset "metric labels: unrouted model names and region names add no series (GW-5)" begin
    # No workers at all: every request below is for a model nothing serves, and the shared-memory
    # fan-out has nobody to ask. The route refresher never scans, so the table stays as set here.
    pool = GW.ClientPool(String[], Dict{String, GW.WorkerClients}())
    routes = GW.DiscoveredRoutes()
    GW.swap_table!(routes, GW.RoutingTable(Dict("m" => ["w0", "w1"])))
    metrics = GW.GatewayMetrics()
    refresher = GW.RouteRefresher(pool, routes, metrics; min_interval = Inf)
    s = _pk_state()
    st = GW.GatewayState(pool, routes, GW.RegisterGate(), metrics, refresher, s)

    infer(name) = try
        GW._gw_infer(_lenc(_LInf.ModelInferRequest(; model_name = name)), st)
    catch e
        e
    end
    @test GW._model_label(st, "m") == "m"
    @test GW._model_label(st, "nope") == GW.UNKNOWN_MODEL_LABEL == "_unknown"

    # The first unrouted name creates the fixed label's series; 10,000 more add none.
    @test infer("first-unknown").code == gRPCServer.StatusCode.NOT_FOUND
    requests = _series_lines(metrics, "gateway_requests_total")
    latency = _series_lines(metrics, "gateway_request_latency_seconds_count")
    @test requests == Set(["rpc=\"ModelInfer\",model=\"_unknown\",status=\"NotFound\""])
    names = unique([_lrandname() for _ in 1:10_000])
    for n in names
        infer(n)
    end
    @test _series_lines(metrics, "gateway_requests_total") == requests
    @test _series_lines(metrics, "gateway_request_latency_seconds_count") == latency
    # An empty or unreadable name is not routed either.
    @test infer("").code == gRPCServer.StatusCode.INVALID_ARGUMENT
    try
        GW._gw_infer(UInt8[0x0a, 0x09, 0x41], st)
    catch
    end
    @test all(l -> occursin("model=\"_unknown\"", l), _series_lines(metrics, "gateway_requests_total"))
    # Every name was counted, under the one label.
    io = IOBuffer()
    GW.expose(io, metrics)
    counted = match(r"gateway_requests_total\{rpc=\"ModelInfer\",model=\"_unknown\",status=\"NotFound\"\} (\S+)", String(take!(io)))
    @test parse(Float64, counted[1]) == length(names) + 1

    # lpt_packing keeps an arrival counter per name it is given, and it is given routed names only.
    @test isempty(@atomic s.arrivals)

    # Shared-memory region names are chosen by each client process: no series carries one.
    reg(n) = GW._gw_shm_register(_lenc(_LInf.SystemSharedMemoryRegisterRequest(; name = n, key = "/k", byte_size = UInt64(64))), st)
    unreg(n) = GW._gw_shm_unregister(_lenc(_LInf.SystemSharedMemoryUnregisterRequest(; name = n)), st)
    reg("first-region")
    unreg("first-region")
    requests = _series_lines(metrics, "gateway_requests_total")
    latency = _series_lines(metrics, "gateway_request_latency_seconds_count")
    @test "rpc=\"SystemSharedMemoryRegister\",model=\"\",status=\"OK\"" in requests
    @test "rpc=\"SystemSharedMemoryUnregister\",model=\"\",status=\"OK\"" in requests
    for n in unique([_lrandname() for _ in 1:10_000])
        reg(n)
        unreg(n)
    end
    @test _series_lines(metrics, "gateway_requests_total") == requests
    @test _series_lines(metrics, "gateway_request_latency_seconds_count") == latency
    @test !any(l -> occursin("first-region", l), requests)
end
