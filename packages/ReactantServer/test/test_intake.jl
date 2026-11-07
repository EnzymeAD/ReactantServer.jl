# Request intake: one client's mistake is refused, or fails, for that client alone, and never costs
# the worker or the other clients. Through the ModelInfer handler, against the mock runtime; uses the
# helpers of test_status_cancel.jl (_SC, _sc_entry, _sc_sched, _sc_body, _sc_call, _sc_code).

# How many times the mock program has run. Counted at file scope: the dispatch loop calls the program
# in the world it was started in, so the function must exist before any loop starts.
const _IK_RUNS = Threads.Atomic{Int}(0)
_ik_program(a) = (Threads.atomic_add!(_IK_RUNS, 1); [a[1] .* a[2]])

_ik_bytes(x::Array{Float32}) = collect(reinterpret(UInt8, vec(x)))
_ik_tensor(name, x::Matrix{Float32}) = _SC.inference.var"ModelInferRequest.InferInputTensor"(;
    name, datatype = "FP32", shape = Int64[reverse(size(x))...]
)

@testset "a request naming an input or a requested output twice is refused (WRK-11)" begin
    sched, ctx = _sc_sched(_sc_entry("m"; fn = _ik_program))
    _SC.start!(sched)
    _IK_RUNS[] = 0
    try
        # The input x twice, with different values. A coalesced dispatch used to run the first copy
        # and a request dispatched alone the last, so the answer depended on the load.
        a, b = reshape(Float32[1, 2], 2, 1), reshape(Float32[5, 7], 2, 1)
        twice = _SC._encode_message(
            _SC.inference.ModelInferRequest(;
                model_name = "m", inputs = [_ik_tensor("x", a), _ik_tensor("x", b)],
                raw_input_contents = [_ik_bytes(a), _ik_bytes(b)]
            )
        )
        r = _sc_call(ctx, twice)
        @test _sc_code(r) == _SCStatus.INVALID_ARGUMENT
        @test r isa _SC.gRPCServer.GRPCError && occursin("input 'x' appears more than once", r.message)
        r = _sc_call(ctx, _sc_body("m"; outputs = ["y", "y"]))
        @test _sc_code(r) == _SCStatus.INVALID_ARGUMENT
        @test r isa _SC.gRPCServer.GRPCError && occursin("requested output 'y' appears more than once", r.message)
        @test _IK_RUNS[] == 0                                 # neither reached the model

        # Validation refuses a duplicate as well, for a request that did not come through the codec.
        dup = _SC.InferRequest("m", String[], [_SC.NamedTensor("x", a), _SC.NamedTensor("x", b)], Int64(0))
        err = try
            _SC._validate_inputs(sched.registry.by_name["m"], dup)
        catch e
            e
        end
        @test _sc_code(err) == _SCStatus.INVALID_ARGUMENT

        # A well-formed request still runs.
        @test _sc_call(ctx, _sc_body("m"; outputs = ["y"])) isa Vector{UInt8}
        @test _IK_RUNS[] == 1
    finally
        _SC.shutdown!(sched)
    end
end
