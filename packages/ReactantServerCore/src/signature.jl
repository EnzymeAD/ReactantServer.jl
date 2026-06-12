# Seam for validating a manifest against the compiled StableHLO signature.
#
# Full signature validation (executable tensor names, dtypes, and fixed shapes must
# match the StableHLO main function) requires the runtime layer to inspect the
# compiled program. The bundle loader takes a validator so that check can be enabled
# later without reshaping the loader. The default is a no-op.

abstract type SignatureValidator end

struct NullSignatureValidator <: SignatureValidator end

validate_against_signature(::NullSignatureValidator, ::Manifest, ::AbstractVector{UInt8}) = nothing
