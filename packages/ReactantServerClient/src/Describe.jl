# What a server says about the models it serves, over ReactantServer's own DescribeModels RPC
# (ControlService, answered by a worker or the gateway). KServe's ModelMetadata gives the tensors
# alone, with -1 for a batch axis and a variable axis alike; this adds what a client needs to build a
# valid request without reading the bundle: the kind of every axis, the compiled batch sizes and
# input shapes, and the sub-models a meta model calls.

"""
    TensorDescription

One input or output of a described model: its `name`, KServe `datatype` string (e.g. `"FP32"`),
and `shape` in Julia column-major order, the order you build arrays in, with `-1` on every batch
or variable axis. `axes` gives the kind of each axis of `shape`: `:fixed`, `:batch`, or
`:variable`.
"""
struct TensorDescription
    name::String
    datatype::String
    shape::Vector{Int}
    axes::Vector{Symbol}
end

"""
    ModelDescription

What [`describe_models`](@ref) reports for one model:

- `name`, `kind` (`:model`, or `:meta` for a Julia pipeline over other models), `description`
- `inputs`, `outputs`: [`TensorDescription`](@ref)s, in declaration order
- `batch_sizes`: the batch sizes the bundle was compiled for, ascending (empty when it declares none)
- `max_batch_size`: the most rows the worker coalesces into one execution, after any configured cap
  (0 for a meta model, or when the server does not know it). [`KServeModel`](@ref) refuses a
  `max_batch_size` below 1, so in that case pass the largest of `batch_sizes`, or 1 when it is empty
- `calls`: a meta model's sub-models, which are internal: request the meta model, not them
- `variants`: the compiled input shapes of a model whose inputs have variable axes, one
  `Dict(input name => Julia column-major shape)` per variant with the batch axis left at `-1`.
  Empty when no input has a variable axis, and when the model's preprocessing transforms its
  inputs (the compiled shapes then do not constrain what a client sends).
"""
struct ModelDescription
    name::String
    kind::Symbol
    description::String
    inputs::Vector{TensorDescription}
    outputs::Vector{TensorDescription}
    batch_sizes::Vector{Int}
    max_batch_size::Int
    calls::Vector{String}
    variants::Vector{Dict{String, Vector{Int}}}
end

function _axis_symbol(k)
    k == control.AxisKind.AXIS_FIXED && return :fixed
    k == control.AxisKind.AXIS_BATCH && return :batch
    k == control.AxisKind.AXIS_VARIABLE && return :variable
    return error("unknown axis kind $k in a DescribeModels response")
end

# The wire is row-major, like every KServe shape; reverse into the Julia order a caller builds in.
function _tensor_description(t)
    length(t.axes) == length(t.shape) ||
        error("tensor '$(t.name)' reports $(length(t.axes)) axis kinds for a rank-$(length(t.shape)) shape")
    return TensorDescription(
        t.name, t.datatype, reverse(Int[Int(s) for s in t.shape]), reverse(Symbol[_axis_symbol(k) for k in t.axes])
    )
end

function _model_description(m)
    return ModelDescription(
        m.name, Symbol(m.kind), m.description,
        TensorDescription[_tensor_description(t) for t in m.inputs],
        TensorDescription[_tensor_description(t) for t in m.outputs],
        Int[Int(b) for b in m.batch_sizes], Int(m.max_batch_size), collect(String, m.calls),
        [Dict(t.name => reverse(Int[Int(s) for s in t.shape]) for t in v.inputs) for v in m.variants]
    )
end

function _describe(chan::gRPCClient.gRPCChannel, names; deadline)
    req = control.DescribeModelsRequest(; names = collect(String, names))
    resp = _call_unary(deadline) do b
        control.ControlService.DescribeModels(chan, req, gRPCClient.gRPCAsync(); deadline = b)
    end
    return ModelDescription[_model_description(m) for m in resp.models]
end

"""
    describe_models(url; names = String[], deadline = 10.0, grpc = gRPCClient.grpc_global_handle())
        -> Vector{ModelDescription}

Ask the server at `url` (the gateway or a worker, `grpc://host:port`) to describe the models in
`names`, or every model it routes requests to when `names` is empty, sorted by name. The
unfiltered list leaves out the internal sub-models of meta models; name one to describe it. Throws
the server's `NOT_FOUND` for a name it does not serve. A server that predates the RPC answers
`UNIMPLEMENTED`. Like every call on the global gRPC handle, it needs [`kserve_init`](@ref) first.
"""
function describe_models(
        url::AbstractString; names = String[], deadline::Real = 10.0,
        grpc::gRPCClient.gRPCCURL = gRPCClient.grpc_global_handle()
    )
    host, port, secure = parse_grpc_url(String(url))
    chan = gRPCClient.gRPCChannel(host, Int(port); grpc = grpc, secure = secure, deadline = deadline)
    return _describe(chan, names; deadline = deadline)
end

"""
    describe_model(model::KServeModel) -> ModelDescription

Describe the model behind `model`, over its own connection and with its deadline. See
[`describe_models`](@ref).
"""
describe_model(model::KServeModel) =
    only(_describe(model.chan, [model_name(model)]; deadline = model.deadline))
