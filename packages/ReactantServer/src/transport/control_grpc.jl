# The ReactantServer ControlService gRPC handlers. Distinct from the KServe data plane, this
# exposes residency and live-policy control to a gateway/control plane. Handlers translate
# between the control protobuf and the scheduler's control functions (set_residency!, set_policy!,
# control_status), which run the actual work on the dispatch thread. Registered on the same server
# and InferContext payload as the inference service (see build_grpc_server).

const _CTRL = control

# Map between the protobuf Residency enum and the Core ResidencyState.
function _to_core_residency(r)
    r == _CTRL.Residency.UNPINNED && return UNPINNED
    r == _CTRL.Residency.PINNED_SYSTEM && return PINNED_SYSTEM
    r == _CTRL.Residency.PINNED_DEVICE && return PINNED_DEVICE
    return _invalid("residency target must be UNPINNED, PINNED_SYSTEM, or PINNED_DEVICE")
end

_from_core_residency(s::ResidencyState) =
    s == UNPINNED ? _CTRL.Residency.UNPINNED :
    s == PINNED_SYSTEM ? _CTRL.Residency.PINNED_SYSTEM : _CTRL.Residency.PINNED_DEVICE

# Run a control action, mapping a thrown error to NOT_FOUND for an unknown model and to
# FAILED_PRECONDITION otherwise (e.g. residency control on a self-managed worker).
function _as_control(f)
    try
        return f()
    catch e
        e isa gRPCServer.GRPCError && rethrow()
        msg = sprint(showerror, e)
        occursin("unknown model", msg) && _not_found(msg)
        throw(gRPCServer.GRPCError(gRPCServer.StatusCode.FAILED_PRECONDITION, msg))
    end
end

function _handle_model_control_status(ctx::InferContext)
    snap = control_status(ctx.sched)
    costs = load_costs(ctx.sched, keys(snap.models))
    nocost = (host_copy = false, upload_seconds = 0.0, read_seconds = 0.0, loads = 0)
    models = [
        _CTRL.ModelStatus(;
            name = String(name),
            residency = _from_core_residency(m.state),
            device_resident = m.device_resident, host_resident = m.host_resident,
            weight_nbytes = m.weight_nbytes, weight = m.weight, queue_depth = m.queue_depth,
            total_compute_seconds = m.total_compute,
            requests_served = UInt64(m.requests_served),
            dispatch_count = UInt64(m.dispatch_count),
            max_batch_size = Int64(m.max_batch_size),
            # Routing metadata: where the batch axis is, and the row counter that turns
            # total_compute_seconds into a cost per ITEM rather than per request. A gateway
            # that routes by work needs both; omitting either silently degrades it to
            # counting requests, which is indistinguishable from working.
            batch_input_name = String(m.batch_input_name),
            batch_axis = Int64(m.batch_axis),
            rows_served = UInt64(m.rows_served),
            # What loading the model here costs, for a gateway pricing a placement change.
            host_copy = get(costs, name, nocost).host_copy,
            upload_seconds = get(costs, name, nocost).upload_seconds,
            read_seconds = get(costs, name, nocost).read_seconds,
            loads = UInt64(max(get(costs, name, nocost).loads, 0))
        )
            for (name, m) in snap.models
    ]
    return _CTRL.ModelControlStatusResponse(;
        residency_mode = (snap.residency_mode == SELF_MANAGED ? "self_managed" : "externally_managed"),
        discipline = lowercase(string(snap.discipline)),
        models = models,
        weight_cache_max_bytes = UInt64(snap.weight_cache_max_bytes),
        host_weight_budget_bytes = UInt64(snap.host_weight_budget_bytes),
        host_cached_bytes = UInt64(snap.host_cached_bytes)
    )
end

function _handle_stage_weights(ctx::InferContext, req)
    rows = _as_control(
        () -> stage_weights!(ctx.sched, collect(String, req.names); hold_seconds = req.hold_seconds > 0 ? req.hold_seconds : 60.0)
    )
    return _CTRL.StageWeightsResponse(;
        models = [
            _CTRL.StagedModel(;
                name = r.name, state = r.state, read = r.read, weight_nbytes = Int64(r.nbytes),
                seconds = Float64(r.seconds), error = r.error
            ) for r in rows
        ]
    )
end

function _handle_set_model_residency(ctx::InferContext, req)
    target = _to_core_residency(req.target)
    new_state = _as_control(() -> set_residency!(ctx.sched, req.name, target))
    return _CTRL.SetModelResidencyResponse(; residency = _from_core_residency(new_state))
end

function _handle_set_model_policy(ctx::InferContext, req)
    return _as_control() do
        set_policy!(ctx.sched, req.name; weight = req.has_weight ? req.weight : nothing)
        _CTRL.SetModelPolicyResponse()
    end
end

function _handle_compact_memory(ctx::InferContext, req)
    return _as_control() do
        reloaded = compact!(ctx.sched; reload_models = collect(String, req.reload_models))
        resident = ctx.sched.weight_cache === nothing ? 0 : weight_cache_stats(ctx.sched.weight_cache).resident_bytes
        _CTRL.CompactMemoryResponse(; reloaded_models = Int64(reloaded), resident_bytes_after = UInt64(resident))
    end
end

function _handle_describe_models(ctx::InferContext, req)
    found = _as_control(() -> describe_models(ctx.sched, collect(String, req.names)))
    return _CTRL.DescribeModelsResponse(;
        models = [
            encode_model_description(_CTRL, d.name, d.manifest; max_batch_size = d.max_batch_size)
                for d in found
        ]
    )
end

# Register the ControlService handlers on an existing gRPC server (the inference server),
# reusing the InferContext payload so both services share the scheduler.
function register_control_service!(server::gRPCServer.GRPCServer)
    # Every control RPC takes the scheduler lock or does residency work, so none runs on the
    # interactive thread (see `_offloaded`).
    register_ControlService!(
        server;
        ModelControlStatus = _offloaded((ctx, req) -> _handle_model_control_status(ctx.payload)),
        SetModelResidency = _offloaded((ctx, req) -> _handle_set_model_residency(ctx.payload, req)),
        SetModelPolicy = _offloaded((ctx, req) -> _handle_set_model_policy(ctx.payload, req)),
        CompactMemory = _offloaded((ctx, req) -> _handle_compact_memory(ctx.payload, req)),
        StageWeights = _offloaded((ctx, req) -> _handle_stage_weights(ctx.payload, req)),
        DescribeModels = _offloaded((ctx, req) -> _handle_describe_models(ctx.payload, req))
    )
    return server
end
