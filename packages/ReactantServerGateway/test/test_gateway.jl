# Integration tests for the pure-Julia gateway. Two mock worker gRPC servers (typed handlers)
# sit behind a real serve_gateway; the tests drive it with a typed gRPCClient and assert routing,
# round-robin, failover, NotFound, SHM fan-out with rollback, and the admin endpoints. Reuses the
# free-port and grpc_call helpers from grpc_helpers.jl.

import gRPCServer
import gRPCClient
import HTTP
import ProtoBuf as TPB

const GWInf = ReactantServerGateway.inference

@testset "peek_batch_size: item count by input name and axis" begin
    enc(msg) = (io = IOBuffer(); TPB.encode(TPB.ProtoEncoder(io), msg); take!(io))
    tensor(name, shape) = GWInf.var"ModelInferRequest.InferInputTensor"(;
        name = name, datatype = "UINT8", shape = Int64[shape...]
    )
    peek = ReactantServerGateway.peek_batch_size

    # A detector-shaped request: `whcn` puts the batch LAST, so position 0 would report the image
    # width. The worker tells the gateway the axis precisely so this cannot be guessed wrong.
    det = enc(
        GWInf.ModelInferRequest(;
            model_name = "det", id = "r1",
            inputs = [tensor("INPUT__0", (1024, 768, 3, 2))],
            raw_input_contents = [rand(UInt8, 512 * 1024)]
        )
    )
    @test peek(det, "INPUT__0", 4) == 2        # the declared batch axis
    @test peek(det, "INPUT__0", 1) == 1024     # what a first-axis assumption would have charged

    # A cross-encoder-shaped request: the FIRST input carries no batch axis at all, and the batch
    # lives on the second one. Matching by name is what makes this work.
    xenc = enc(
        GWInf.ModelInferRequest(;
            model_name = "xe",
            inputs = [
                tensor("query", (37,)),
                tensor("keys", (12, 37)),
                tensor("key_lens", (12,)),
            ]
        )
    )
    @test peek(xenc, "keys", 1) == 12
    @test peek(xenc, "key_lens", 1) == 12

    # Inputs may arrive in any order: KServe tensors are name-addressed.
    shuffled = enc(
        GWInf.ModelInferRequest(;
            model_name = "xe",
            inputs = [
                tensor("key_lens", (5,)),
                tensor("keys", (5, 9)),
                tensor("query", (9,)),
            ]
        )
    )
    @test peek(shuffled, "keys", 1) == 5

    # Inline contents rather than raw: the shape is field 3 and the data field 5 within the tensor,
    # so it is still reached without decoding the payload.
    inline = GWInf.var"ModelInferRequest.InferInputTensor"(;
        name = "x", datatype = "FP32", shape = Int64[4, 8],
        contents = GWInf.InferTensorContents(; fp32_contents = Float32[1:32;])
    )
    @test peek(enc(GWInf.ModelInferRequest(; model_name = "m", inputs = [inline])), "x", 2) == 8

    # Degenerate cases all report 0, which the caller charges as a single item: no such input, an
    # axis past the end of the shape, an empty name, and a non-positive axis.
    @test peek(det, "nope", 1) == 0
    @test peek(det, "INPUT__0", 9) == 0
    @test peek(det, "", 1) == 0
    @test peek(det, "INPUT__0", 0) == 0

    # model_name and id still come out of their own peek, untouched by any of this.
    @test ReactantServerGateway.peek_model_name_and_id(det) == ("det", "r1")
end

@testset "peek_uses_shared_memory: any input or requested output in a shared-memory region" begin
    enc(msg) = (io = IOBuffer(); TPB.encode(TPB.ProtoEncoder(io), msg); take!(io))
    param(s) = GWInf.InferParameter(; parameter_choice = TPB.OneOf(:string_param, s))
    peek = ReactantServerGateway.peek_uses_shared_memory
    tensor(params) = GWInf.var"ModelInferRequest.InferInputTensor"(;
        name = "x", datatype = "FP32", shape = Int64[2], parameters = params,
        contents = GWInf.InferTensorContents(; fp32_contents = Float32[1, 2])
    )
    out(params) = GWInf.var"ModelInferRequest.InferRequestedOutputTensor"(; name = "y", parameters = params)
    shm = Dict("shared_memory_region" => param("r"), "shared_memory_byte_size" => GWInf.InferParameter(; parameter_choice = TPB.OneOf(:int64_param, Int64(8))))
    other = Dict("binary_data" => param("true"))
    @test !peek(enc(GWInf.ModelInferRequest(; model_name = "m")))
    @test !peek(enc(GWInf.ModelInferRequest(; model_name = "m", inputs = [tensor(other)], outputs = [out(other)])))
    @test peek(enc(GWInf.ModelInferRequest(; model_name = "m", inputs = [tensor(other), tensor(shm)])))
    @test peek(enc(GWInf.ModelInferRequest(; model_name = "m", inputs = [tensor(other)], outputs = [out(shm)])))
    # A request-level parameter of the same name is not a tensor in shared memory.
    @test !peek(enc(GWInf.ModelInferRequest(; model_name = "m", parameters = Dict("shared_memory_region" => param("r")))))
    # A body that cannot be read that far counts as shared memory: never run it twice.
    @test peek(enc(GWInf.ModelInferRequest(; model_name = "m", inputs = [tensor(other)]))[1:(end - 3)])
end

@testset "invoke_infer: a transport failure is UNAVAILABLE" begin
    GW = ReactantServerGateway
    body = UInt8[0x0a, 0x01, 0x61]
    # Nothing listens: libcurl's connect error comes back from gRPCClient as INTERNAL, the code a
    # worker sends for its own failures. It must fail over, so it is UNAVAILABLE here.
    port = grpc_free_port()
    grpc = gRPCClient.gRPCCURL(; sticky = true)
    chan = gRPCClient.gRPCChannel("127.0.0.1", port; grpc = grpc, deadline = 5)
    wc = GW.WorkerClients("127.0.0.1:$port", grpc, chan, 5.0, 5.0)
    err = try
        GW.invoke_infer(wc, body)
    catch e
        e
    end
    @test err isa gRPCClient.gRPCServiceCallException
    @test err.grpc_status == gRPCClient.GRPC_UNAVAILABLE
    @test GW.client_status(err) == GW.STATUS_UNAVAILABLE
    gRPCClient.grpc_shutdown(grpc)
end

mutable struct MockWorker
    name::String
    models::Vector{String}     # models this worker reports READY via RepositoryIndex
    fail_infer::Bool
    fail_shm::Bool
    fail_exhausted::Bool        # shed ModelInfer with RESOURCE_EXHAUSTED (worker concurrency cap)
    empty_infer::Bool           # answer ModelInfer with an empty body (a call aborted by a reset)
    internal::Vector{String}    # models it serves but does not list (a meta model's sub-models)
    metadata::Symbol            # the metadata RPCs: :ok, :down (UNAVAILABLE), :old (UNIMPLEMENTED)
    fail_invalid::Bool          # reject ModelInfer with INVALID_ARGUMENT (e.g. an unknown requested output)
    infer_calls::Int            # ModelInfer calls received, so a test can see whether the gateway failed over
    fail_code::Any              # reject ModelInfer with this gRPCServer.StatusCode, or nothing
end
MockWorker(name, models, fail_infer, fail_shm, fail_exhausted) =
    MockWorker(name, models, fail_infer, fail_shm, fail_exhausted, false, String[], :ok, false, 0, nothing)

# The metadata RPCs' failure modes, and whether the mock serves `name` at all.
function _mock_metadata_check(w::MockWorker)
    w.metadata === :down && throw(gRPCServer.GRPCError(gRPCServer.StatusCode.UNAVAILABLE, "mock $(w.name) down"))
    w.metadata === :old && throw(gRPCServer.GRPCError(gRPCServer.StatusCode.UNIMPLEMENTED, "Method not found"))
    return nothing
end
_mock_serves(w::MockWorker, name) = name in w.models || name in w.internal
_mock_not_found(w::MockWorker, names) =
    throw(gRPCServer.GRPCError(gRPCServer.StatusCode.NOT_FOUND, "unknown model: $(join(names, ", ")) on $(w.name)"))

# A description that says which worker wrote it, so a test can see whose answer the gateway kept.
_mock_description(w::MockWorker, name) =
    ReactantServerGateway.control.ModelDescription(; name = name, kind = "model", description = w.name)

function _mock_server(worker::MockWorker)
    server = gRPCServer.GRPCServerHTTPJl("127.0.0.1", 1; context = worker)
    server.port = 0   # OS-assigned ephemeral port; HTTP.port(server) reports it after start!
    ReactantServerGateway.register_GRPCInferenceService!(
        server;
        ServerReady = (ctx, req) -> GWInf.ServerReadyResponse(; ready = true),
        RepositoryIndex = (ctx, req) -> GWInf.RepositoryIndexResponse(;
            models = [
                GWInf.var"RepositoryIndexResponse.ModelIndex"(; name = m, version = "", state = "READY", reason = "")
                    for m in ctx.payload.models
            ]
        ),
        ModelInfer = (ctx, req) -> begin
            w = ctx.payload
            w.infer_calls += 1
            w.fail_infer && throw(gRPCServer.GRPCError(gRPCServer.StatusCode.UNAVAILABLE, "mock $(w.name) down"))
            w.fail_invalid && throw(gRPCServer.GRPCError(gRPCServer.StatusCode.INVALID_ARGUMENT, "requested output 'x' is not produced by the model"))
            w.fail_exhausted && throw(gRPCServer.GRPCError(gRPCServer.StatusCode.RESOURCE_EXHAUSTED, "mock $(w.name) at capacity"))
            w.fail_code === nothing || throw(gRPCServer.GRPCError(w.fail_code, "mock $(w.name) failed with $(w.fail_code)"))
            w.empty_infer && return GWInf.ModelInferResponse()   # every field default: encodes to 0 bytes
            # A real worker NOT_FOUNDs a model it no longer serves; mirror that so the gateway's
            # unloaded-model handling (route refresh on worker NOT_FOUND) can be exercised.
            (req.model_name in w.models) ||
                throw(gRPCServer.GRPCError(gRPCServer.StatusCode.NOT_FOUND, "model $(req.model_name) not on $(w.name)"))
            GWInf.ModelInferResponse(; model_name = w.name, id = req.id)
        end,
        SystemSharedMemoryRegister = (ctx, req) -> begin
            ctx.payload.fail_shm && throw(gRPCServer.GRPCError(gRPCServer.StatusCode.FAILED_PRECONDITION, "mock shm fail"))
            GWInf.SystemSharedMemoryRegisterResponse()
        end,
        SystemSharedMemoryUnregister = (ctx, req) -> GWInf.SystemSharedMemoryUnregisterResponse(),
        ModelMetadata = (ctx, req) -> begin
            w = ctx.payload
            _mock_metadata_check(w)
            _mock_serves(w, req.name) || _mock_not_found(w, [req.name])
            GWInf.ModelMetadataResponse(; name = req.name, platform = w.name)
        end,
    )
    ReactantServerGateway.register_ControlService!(
        server;
        DescribeModels = (ctx, req) -> begin
            w = ctx.payload
            _mock_metadata_check(w)
            names = isempty(req.names) ? sort(w.models) : sort(unique(req.names))
            unknown = filter(n -> !_mock_serves(w, n), names)
            isempty(unknown) || _mock_not_found(w, unknown)
            ReactantServerGateway.control.DescribeModelsResponse(; models = [_mock_description(w, n) for n in names])
        end,
    )
    return server
end

# Start a mock worker on an OS-assigned port and block until it actually answers gRPC
# (ServerReady), so the gateway's first synchronous discovery round always probes a live
# worker. start! is async and the first request on a cold CI runner can be slow, so a
# readiness poll beats a fixed sleep: the gateway must never discover a half-started mock.
function _start_mock(worker::MockWorker; timeout::Real = 30.0)
    server = _mock_server(worker)
    gRPCServer.start!(server)
    # HTTP.port reads the listener's bound address; with an ephemeral port it is only
    # meaningful once start! has actually bound, so poll for it first.
    deadline = time() + timeout
    port = 0
    while time() < deadline
        port = try
            Int(HTTP.port(server))
        catch
            0
        end
        port > 0 && break
        sleep(0.1)
    end
    port > 0 || error("mock worker $(worker.name) never bound a port")
    while time() < deadline
        try
            GWInf.GRPCInferenceService.ServerReady("127.0.0.1", port, GWInf.ServerReadyRequest())
            return server, port
        catch
            sleep(0.1)
        end
    end
    error("mock worker $(worker.name) did not become ready on port $port")
end

# Send a typed ModelInfer through the gateway and return the response (or rethrow).
_infer(port, model) = grpc_call(
    GWInf.GRPCInferenceService.ModelInfer, port, GWInf.ModelInferRequest(; model_name = model)
)

function _http_get(port, path)
    try
        return HTTP.get("http://127.0.0.1:$port$path"; retry = false, status_exception = false)
    catch e
        return e
    end
end

@testset "gateway" begin
    # worker0 serves replicated + only0; worker1 serves replicated. The gateway discovers this
    # from each worker's RepositoryIndex rather than from any config.
    w0 = MockWorker("worker0", ["replicated", "only0"], false, false, false)
    w1 = MockWorker("worker1", ["replicated"], false, false, false)
    # OS-assigned worker ports: the mock binds an ephemeral port (no free-port race), and the
    # readiness poll above guarantees it is serving before the gateway starts.
    s0, p0 = _start_mock(w0)
    s1, p1 = _start_mock(w1)

    gatewayfile = tempname() * ".yaml"
    write(
        gatewayfile, """
        listen:
          grpc: "127.0.0.1:0"
          metrics: "127.0.0.1:0"
        endpoints:
          - "127.0.0.1:$p0"
          - "127.0.0.1:$p1"
        """
    )

    gw = ReactantServerGateway.serve_gateway(gatewayfile; blocking = false)
    gw_port, admin_port = ReactantServerGateway.bound_ports(gw)

    # Wait for the first discovery round to populate routes before asserting.
    routed = false
    for _ in 1:40
        try
            _infer(gw_port, "only0")
            routed = true
            break
        catch
            sleep(0.1)
        end
    end
    @test routed

    try
        @testset "routing" begin
            r = _infer(gw_port, "only0")
            @test r.model_name == "worker0"
        end

        @testset "round-robin across replicas" begin
            seen = Set{String}()
            for _ in 1:4
                push!(seen, _infer(gw_port, "replicated").model_name)
            end
            @test seen == Set(["worker0", "worker1"])
        end

        @testset "unknown model is NotFound" begin
            err = try
                _infer(gw_port, "nope")
                nothing
            catch e
                e
            end
            @test err isa gRPCClient.gRPCServiceCallException
            @test err.grpc_status == gRPCClient.GRPC_NOT_FOUND
        end

        @testset "failover to healthy replica" begin
            w0.fail_infer = true
            for _ in 1:4
                @test _infer(gw_port, "replicated").model_name == "worker1"
            end
            # only0 has no healthy replica -> error surfaced
            err = try
                _infer(gw_port, "only0")
                nothing
            catch e
                e
            end
            @test err isa gRPCClient.gRPCServiceCallException
            w0.fail_infer = false
        end

        @testset "an empty worker response fails over instead of passing as a success" begin
            # gRPCClient before 1.1.2 completed a call aborted by reset_clients! with an empty body.
            # A real worker always sets model_name, so the gateway treats empty as unavailable.
            w0.empty_infer = true
            for _ in 1:4
                @test _infer(gw_port, "replicated").model_name == "worker1"
            end
            err = try
                _infer(gw_port, "only0")
                nothing
            catch e
                e
            end
            @test err isa gRPCClient.gRPCServiceCallException
            @test err.grpc_status == gRPCClient.GRPC_UNAVAILABLE
            w0.empty_infer = false
        end

        @testset "worker overload shed surfaces as RESOURCE_EXHAUSTED" begin
            # only0 is served solely by w0, so the gateway must route there. A worker shedding at
            # its concurrency cap must reach the client as RESOURCE_EXHAUSTED (a retryable overload
            # signal), not be remapped to FAILED_PRECONDITION.
            w0.fail_exhausted = true
            err = try
                _infer(gw_port, "only0")
                nothing
            catch e
                e
            end
            @test err isa gRPCClient.gRPCServiceCallException
            @test err.grpc_status == gRPCClient.GRPC_RESOURCE_EXHAUSTED
            w0.fail_exhausted = false
        end

        @testset "a worker's INVALID_ARGUMENT reaches the client without failover" begin
            # Every replica would reject the same request, so trying the next one only doubles the
            # work, and folding it into UNAVAILABLE invites the client to retry a request that
            # cannot succeed.
            w0.fail_invalid = w1.fail_invalid = true
            w0.infer_calls = w1.infer_calls = 0
            err = try
                _infer(gw_port, "replicated")
                nothing
            catch e
                e
            end
            @test err isa gRPCClient.gRPCServiceCallException
            @test err.grpc_status == gRPCClient.GRPC_INVALID_ARGUMENT
            @test occursin("is not produced by the model", err.message)
            @test w0.infer_calls + w1.infer_calls == 1
            w0.fail_invalid = w1.fail_invalid = false
        end

        @testset "a worker's status reaches the client verbatim; only NOT_FOUND and UNAVAILABLE fail over" begin
            # Before, the gateway folded DEADLINE_EXCEEDED, INTERNAL, CANCELLED and the rest into
            # UNAVAILABLE and failed over, so a deterministic failure ran on every replica and then
            # again on each client retry, and a client never saw that its deadline had passed.
            S = gRPCServer.StatusCode
            calls() = w0.infer_calls + w1.infer_calls
            outcome(model) = try
                _infer(gw_port, model)
                nothing
            catch e
                e
            end
            try
                for code in (
                        S.DEADLINE_EXCEEDED, S.INTERNAL, S.CANCELLED, S.UNKNOWN, S.ABORTED,
                        S.UNIMPLEMENTED, S.OUT_OF_RANGE, S.PERMISSION_DENIED, S.DATA_LOSS,
                        S.ALREADY_EXISTS, S.UNAUTHENTICATED, S.FAILED_PRECONDITION,
                        S.RESOURCE_EXHAUSTED,
                    )
                    w0.fail_code = w1.fail_code = code
                    w0.infer_calls = w1.infer_calls = 0
                    err = outcome("replicated")
                    @test err isa gRPCClient.gRPCServiceCallException
                    @test err.grpc_status == Int(code)
                    @test calls() == 1
                end
                for code in (S.NOT_FOUND, S.UNAVAILABLE)
                    w0.fail_code = w1.fail_code = code
                    w0.infer_calls = w1.infer_calls = 0
                    err = outcome("replicated")
                    @test err isa gRPCClient.gRPCServiceCallException
                    @test err.grpc_status == Int(code)
                    @test calls() == 2
                end
                # A replica that is merely unavailable is skipped for a healthy one.
                w0.fail_code, w1.fail_code = S.UNAVAILABLE, nothing
                @test all(_ -> outcome("replicated") === nothing, 1:4)
            finally
                w0.fail_code = w1.fail_code = nothing
            end
        end

        @testset "a request that uses shared memory is never retried on another worker" begin
            # The failed worker may still be running it, and a second worker writing the same output
            # slot, or reading an input slot the client has reused, would corrupt the client's data.
            S = gRPCServer.StatusCode
            shm_params() = Dict(
                "shared_memory_region" => GWInf.InferParameter(; parameter_choice = TPB.OneOf(:string_param, "client-slot-0"))
            )
            in_shm = GWInf.ModelInferRequest(;
                model_name = "replicated",
                inputs = [GWInf.var"ModelInferRequest.InferInputTensor"(; name = "x", datatype = "FP32", shape = Int64[1], parameters = shm_params())]
            )
            out_shm = GWInf.ModelInferRequest(;
                model_name = "replicated",
                outputs = [GWInf.var"ModelInferRequest.InferRequestedOutputTensor"(; name = "y", parameters = shm_params())]
            )
            plain = GWInf.ModelInferRequest(; model_name = "replicated")
            calls() = w0.infer_calls + w1.infer_calls
            outcome(req) = try
                grpc_call(GWInf.GRPCInferenceService.ModelInfer, gw_port, req)
                nothing
            catch e
                e
            end
            try
                for code in (S.UNAVAILABLE, S.NOT_FOUND)
                    w0.fail_code = w1.fail_code = code
                    for req in (in_shm, out_shm)
                        w0.infer_calls = w1.infer_calls = 0
                        err = outcome(req)
                        @test err isa gRPCClient.gRPCServiceCallException
                        @test err.grpc_status == gRPCClient.GRPC_UNAVAILABLE
                        @test occursin("shared memory", err.message)
                        @test calls() == 1
                    end
                    w0.infer_calls = w1.infer_calls = 0
                    @test outcome(plain).grpc_status == Int(code)   # without shared memory it fails over
                    @test calls() == 2
                end
                # A worker's own verdict on a shared-memory request still reaches the client as is
                # (here a stale registration, which the client answers by re-registering).
                w0.fail_code = w1.fail_code = S.FAILED_PRECONDITION
                @test outcome(in_shm).grpc_status == gRPCClient.GRPC_FAILED_PRECONDITION
                w0.fail_code = w1.fail_code = nothing
                @test outcome(in_shm) === nothing
            finally
                w0.fail_code = w1.fail_code = nothing
            end
        end

        @testset "SHM register fan-out and rollback" begin
            reg(name) = grpc_call(
                GWInf.GRPCInferenceService.SystemSharedMemoryRegister, gw_port, GWInf.SystemSharedMemoryRegisterRequest(; name = name)
            )
            # all workers ok
            @test reg("region-ok") isa GWInf.SystemSharedMemoryRegisterResponse
            # one worker fails -> FailedPrecondition
            w1.fail_shm = true
            err = try
                reg("region-bad")
                nothing
            catch e
                e
            end
            @test err isa gRPCClient.gRPCServiceCallException
            @test err.grpc_status == gRPCClient.GRPC_FAILED_PRECONDITION
            w1.fail_shm = false
        end

        @testset "metadata RPCs merge every worker's answer" begin
            Ctl = ReactantServerGateway.control
            describe(names...) = grpc_call(
                Ctl.ControlService.DescribeModels, gw_port, Ctl.DescribeModelsRequest(; names = collect(String, names))
            ).models
            metadata(name) = grpc_call(
                GWInf.GRPCInferenceService.ModelMetadata, gw_port, GWInf.ModelMetadataRequest(; name = name)
            )
            index() = grpc_call(
                GWInf.GRPCInferenceService.RepositoryIndex, gw_port, GWInf.RepositoryIndexRequest()
            ).models
            status_of(f) = try
                f(); nothing
            catch e
                e isa gRPCClient.gRPCServiceCallException ? e.grpc_status : rethrow()
            end
            w1.internal = ["stage"]
            try
                # The union over both workers, sorted; worker0 is first in pool order, so its
                # description of the replicated model is the one kept.
                @test [(m.name, m.description) for m in describe()] ==
                    [("only0", "worker0"), ("replicated", "worker0")]
                @test [m.name for m in index()] == ["only0", "replicated"]
                # An internal sub-model is left out of the full list but described when named,
                # by the worker that has it, and KServe metadata for it reaches that worker too.
                @test [(m.name, m.description) for m in describe("stage", "only0")] ==
                    [("only0", "worker0"), ("stage", "worker1")]
                @test metadata("stage").platform == "worker1"
                @test metadata("only0").platform == "worker0"
                @test status_of(() -> describe("only0", "nope")) == gRPCClient.GRPC_NOT_FOUND
                @test status_of(() -> metadata("nope")) == gRPCClient.GRPC_NOT_FOUND

                # One worker down: the other's answer still comes back, and metadata for a model
                # only the down worker has is UNAVAILABLE rather than NOT_FOUND.
                w0.metadata = :down
                @test [(m.name, m.description) for m in describe()] == [("replicated", "worker1")]
                @test metadata("replicated").platform == "worker1"
                @test status_of(() -> metadata("only0")) == gRPCClient.GRPC_UNAVAILABLE
                @test status_of(() -> describe("only0")) == gRPCClient.GRPC_NOT_FOUND
                # Every worker down is UNAVAILABLE; every worker too old to have the RPC says so.
                w1.metadata = :down
                @test status_of(describe) == gRPCClient.GRPC_UNAVAILABLE
                w0.metadata = w1.metadata = :old
                @test status_of(describe) == gRPCClient.GRPC_UNIMPLEMENTED
            finally
                w0.metadata = w1.metadata = :ok
                w1.internal = String[]
            end
        end

        @testset "admin endpoints" begin
            @test _http_get(admin_port, "/healthz").status == 200
            # readiness flips to 200 once a worker reports ready
            ready = false
            for _ in 1:20
                _http_get(admin_port, "/readyz").status == 200 && (ready = true; break)
                sleep(0.25)
            end
            @test ready
            metrics = _http_get(admin_port, "/metrics")
            @test metrics.status == 200
            body = String(metrics.body)
            @test occursin("gateway_requests_total", body)
            @test occursin("gateway_worker_ready", body)
            @test occursin("gateway_routing_table_size", body)
        end

        @testset "on-demand refresh routes a newly loaded model" begin
            # 'lazy' is unknown to every worker initially, so the gateway answers NOT_FOUND.
            err = try
                _infer(gw_port, "lazy"); nothing
            catch e
                e
            end
            @test err isa gRPCClient.gRPCServiceCallException
            @test err.grpc_status == gRPCClient.GRPC_NOT_FOUND
            # Load it on worker0. Past the refresher's min_interval the next request rescans on
            # demand and routes it, well before the 10s health tick.
            push!(w0.models, "lazy")
            sleep(1.1)
            @test _infer(gw_port, "lazy").model_name == "worker0"
        end

        @testset "on-demand refresh drops an unloaded model" begin
            @test ReactantServerGateway.pick(gw.routes, "only0") !== nothing   # routed before unload
            filter!(!=("only0"), w0.models)                                    # unload on the worker
            # A request to the stale route gets NOT_FOUND from the worker, which kicks an async
            # refresh; within a couple of seconds the route is dropped from the gateway's table.
            dropped = false
            for _ in 1:40
                try
                    _infer(gw_port, "only0")
                catch
                end
                if ReactantServerGateway.pick(gw.routes, "only0") === nothing
                    dropped = true
                    break
                end
                sleep(0.2)
            end
            @test dropped
        end
    finally
        ReactantServerGateway.stop!(gw)
        close(s0)
        close(s1)
        rm(gatewayfile; force = true)
    end
end

@testset "env-only gateway config (no gateway.yml)" begin
    # The node supervisor starts an embedded gateway with no config file: defaults plus
    # REACTANT_GATEWAY_* environment variables, with the endpoint list synthesized into
    # REACTANT_GATEWAY_WORKERS.
    withenv(
        "REACTANT_GATEWAY_WORKERS" => "127.0.0.1:8080,127.0.0.1:8081",
        "REACTANT_GATEWAY_LISTEN_GRPC" => "0.0.0.0:9001"
    ) do
        cfg = ReactantServerGateway.load_gateway(nothing)
        @test cfg.workers == ["127.0.0.1:8080", "127.0.0.1:8081"]
        @test cfg.listen_grpc == "0.0.0.0:9001"
        @test cfg.listen_metrics == "0.0.0.0:8002"       # default
        # gRPC message-size limits default to 512 MiB and honor the env override.
        @test cfg.max_recv_msg_bytes == 512 * 1024 * 1024
        @test cfg.max_send_msg_bytes == 512 * 1024 * 1024
    end

    withenv(
        "REACTANT_GATEWAY_WORKERS" => "127.0.0.1:8080",
        "REACTANT_GATEWAY_GRPC_MAX_RECV_MSG_BYTES" => "4096"
    ) do
        cfg = ReactantServerGateway.load_gateway(nothing)
        @test cfg.max_recv_msg_bytes == 4096
        @test cfg.max_send_msg_bytes == 512 * 1024 * 1024   # unset -> default
    end

    # Without an endpoint list the env-only path keeps the existing guard.
    withenv("REACTANT_GATEWAY_WORKERS" => nothing) do
        @test_throws ReactantServerCore.ConfigError ReactantServerGateway.load_gateway(nothing)
    end

    # An explicit path that does not exist still fails loudly.
    @test_throws ReactantServerCore.ConfigError ReactantServerGateway.load_gateway(tempname())
end

@testset "metrics aggregation" begin
    @testset "merge_expositions" begin
        gw = """
        # HELP gateway_requests_total Count of gateway RPCs.
        # TYPE gateway_requests_total counter
        gateway_requests_total{rpc="ModelInfer",model="m",status="OK"} 2
        """
        w0 = """
        # HELP worker_dispatch_total Total dispatches per model.
        # TYPE worker_dispatch_total counter
        worker_dispatch_total{worker="worker0",gpu="0",model="m"} 3
        # TYPE worker_request_latency_seconds histogram
        worker_request_latency_seconds_bucket{worker="worker0",model="m",le="+Inf"} 1
        worker_request_latency_seconds_count{worker="worker0",model="m"} 1
        """
        w1 = """
        # HELP worker_dispatch_total Total dispatches per model.
        # TYPE worker_dispatch_total counter
        worker_dispatch_total{worker="worker1",gpu="1",model="m"} 5
        """
        merged = ReactantServerGateway.merge_expositions([gw, w0, w1])
        # One header per family, both workers' samples grouped under it.
        @test count("# TYPE worker_dispatch_total counter", merged) == 1
        @test occursin("worker_dispatch_total{worker=\"worker0\",gpu=\"0\",model=\"m\"} 3", merged)
        @test occursin("worker_dispatch_total{worker=\"worker1\",gpu=\"1\",model=\"m\"} 5", merged)
        @test occursin("gateway_requests_total{rpc=\"ModelInfer\",model=\"m\",status=\"OK\"} 2", merged)
        # Histogram suffix lines stay under their family header.
        i_type = findfirst("# TYPE worker_request_latency_seconds histogram", merged)
        i_count = findfirst("worker_request_latency_seconds_count", merged)
        @test i_type !== nothing && i_count !== nothing && first(i_type) < first(i_count)
        # The worker0 block's samples come before any re-declaration would; exposition parses
        # as one block per family (no duplicate headers anywhere).
        @test count("# HELP worker_dispatch_total", merged) == 1
    end

    @testset "aggregated /metrics endpoint" begin
        # Two fake workers serving pre-tagged expositions, one dead endpoint.
        function fake(text)
            s = HTTP.serve!("127.0.0.1", 0) do req
                HTTP.Response(200; body = text)
            end
            return s, "127.0.0.1:$(HTTP.port(s))"
        end
        s0, ep0 = fake("worker_models_loaded{worker=\"worker0\",gpu=\"0\"} 1\n")
        s1, ep1 = fake("worker_models_loaded{worker=\"worker1\",gpu=\"1\"} 1\n")
        dead = "127.0.0.1:$(grpc_free_port())"
        metrics = ReactantServerGateway.GatewayMetrics()
        admin = ReactantServerGateway.start_admin(
            metrics, "127.0.0.1:0";
            worker_metrics = [ep0, ep1, dead]
        )
        admin_port = HTTP.port(admin.server)
        try
            body = String(HTTP.get("http://127.0.0.1:$admin_port/metrics"; retry = false).body)
            @test occursin("worker_models_loaded{worker=\"worker0\",gpu=\"0\"} 1", body)
            @test occursin("worker_models_loaded{worker=\"worker1\",gpu=\"1\"} 1", body)
            @test count("# TYPE gateway_worker_metrics_up gauge", body) == 1
            @test occursin("gateway_worker_metrics_up{worker=\"$ep0\"} 1", body)
            @test occursin("gateway_worker_metrics_up{worker=\"$dead\"} 0", body)
        finally
            close(admin.server)
            close(s0)
            close(s1)
        end
    end

    @testset "the event-loop monitor sees the thread held, and counts a stall" begin
        metrics = ReactantServerGateway.GatewayMetrics()
        m = ReactantServerGateway.LoopMonitor(metrics; interval = 0.02, warn_after = 0.2)
        ReactantServerGateway.start_loop_monitor!(m)
        sleep(0.2)                                   # free ticks: each wakes about on time
        t = time()
        while time() - t < 0.4                       # hold the thread without yielding
        end
        sleep(0.2)                                   # let the monitor wake and record it
        ReactantServerGateway.stop_loop_monitor!(m)
        io = IOBuffer()
        ReactantServerGateway.expose(io, metrics)
        body = String(take!(io))
        # The held tick is a stall; a first tick on a cold process can be one too (it compiles).
        stalls = parse(Float64, match(r"^gateway_event_loop_stalls_total (\S+)$"m, body)[1])
        @test stalls >= 1
        count_ = parse(Float64, match(r"^gateway_event_loop_lag_seconds_count (\S+)$"m, body)[1])
        under = parse(Float64, match(r"^gateway_event_loop_lag_seconds_bucket\{le=\"0.256\"\} (\S+)$"m, body)[1])
        @test count_ >= 2 && under < count_          # some ticks were on time, at least one 0.4 s late
    end

    @testset "latency buckets reach past a minute" begin
        # A 50 s request lands in a finite bucket (and matches the worker's buckets), not in +Inf.
        metrics = ReactantServerGateway.GatewayMetrics()
        ReactantServerGateway.observe_request!(metrics, "ModelInfer", "m", 50.0)
        io = IOBuffer()
        ReactantServerGateway.expose(io, metrics)
        body = String(take!(io))
        @test occursin("gateway_request_latency_seconds_bucket{rpc=\"ModelInfer\",model=\"m\",le=\"32.768\"} 0", body)
        @test occursin("gateway_request_latency_seconds_bucket{rpc=\"ModelInfer\",model=\"m\",le=\"65.536\"} 1", body)
        @test occursin("gateway_request_latency_seconds_bucket{rpc=\"ModelInfer\",model=\"m\",le=\"131.072\"} 1", body)
    end

    @testset "metrics endpoints config" begin
        withenv(
            "REACTANT_GATEWAY_WORKERS" => "127.0.0.1:8080",
            "REACTANT_GATEWAY_WORKER_METRICS" => "127.0.0.1:9100, 127.0.0.1:9101"
        ) do
            cfg = ReactantServerGateway.load_gateway(nothing)
            @test cfg.worker_metrics == ["127.0.0.1:9100", "127.0.0.1:9101"]
        end
        withenv(
            "REACTANT_GATEWAY_WORKERS" => "127.0.0.1:8080",
            "REACTANT_GATEWAY_WORKER_METRICS" => nothing
        ) do
            @test ReactantServerGateway.load_gateway(nothing).worker_metrics == String[]
        end
    end
end
