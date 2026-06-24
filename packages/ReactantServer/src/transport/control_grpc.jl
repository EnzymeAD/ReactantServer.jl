# The ReactantServer ControlService gRPC handlers. Distinct from the KServe data plane, this
# exposes residency and live-policy control to a gateway/control plane. Handlers translate
# between the control protobuf and the scheduler's control functions (set_residency!, set_policy!,
# control_status), which run the actual work on the dispatch thread. Registered on the same router
# and InferContext payload as the inference service (see build_grpc_router).

const _CTRL = ReactantServerCore.control

# Map between the protobuf Residency enum and the Core ResidencyState.
function _to_core_residency(r)
    r == _CTRL.Residency.UNPINNED && return UNPINNED
    r == _CTRL.Residency.PINNED_SYSTEM && return PINNED_SYSTEM
    r == _CTRL.Residency.PINNED_DEVICE && return PINNED_DEVICE
    _invalid("residency target must be UNPINNED, PINNED_SYSTEM, or PINNED_DEVICE")
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
        e isa _G.gRPCServiceCallException && rethrow()
        msg = sprint(showerror, e)
        occursin("unknown model", msg) && _not_found(msg)
        throw(_G.gRPCServiceCallException(_G.GRPC_FAILED_PRECONDITION, msg))
    end
end

function _handle_model_control_status(ctx::InferContext)
    snap = control_status(ctx.sched)
    models = [_CTRL.ModelStatus(; name = String(name),
                  residency = _from_core_residency(m.state),
                  device_resident = m.device_resident, host_resident = m.host_resident,
                  weight_nbytes = m.weight_nbytes, weight = m.weight, queue_depth = m.queue_depth,
                  total_compute_seconds = m.total_compute,
                  requests_served = UInt64(m.requests_served),
                  dispatch_count = UInt64(m.dispatch_count),
                  max_batch_size = Int64(m.max_batch_size))
              for (name, m) in snap.models]
    return _CTRL.ModelControlStatusResponse(;
        residency_mode = (snap.residency_mode == SELF_MANAGED ? "self_managed" : "externally_managed"),
        discipline = lowercase(string(snap.discipline)),
        models = models,
        weight_cache_max_bytes = UInt64(snap.weight_cache_max_bytes))
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

# Register the ControlService handlers on an existing router (the inference router), reusing the
# InferContext payload so both services share the scheduler.
function register_control_service!(router)
    register_ControlService!(router;
        ModelControlStatus = (req, ctx) -> _handle_model_control_status(ctx.payload),
        SetModelResidency  = (req, ctx) -> _handle_set_model_residency(ctx.payload, req),
        SetModelPolicy     = (req, ctx) -> _handle_set_model_policy(ctx.payload, req),
        CompactMemory      = (req, ctx) -> _handle_compact_memory(ctx.payload, req),
    )
    return router
end
