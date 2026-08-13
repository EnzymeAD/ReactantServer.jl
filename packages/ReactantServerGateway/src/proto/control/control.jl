# Thin wrapper. The generated code names the module after the proto package
# (reactant_control); the repo has always exposed it as `control`. The wrapper
# is the only hand-authored part; reactant_control_v1_pb.jl is generated output
# (byte-stable; regenerate per the grpcserver-jl-dev skill) and is module-name
# independent (only the service paths /reactant_control.{Control,GatewayControl}Service/... appear).
module control

include("reactant_control_v1_pb.jl")

end # module control
