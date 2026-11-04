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
        finally
            shutdown!(sched)
        end
    end
end
