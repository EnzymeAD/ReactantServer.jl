# Canonical dtype mapping shared across the server.
#
# One enum is the single source of truth; the maps translate between the manifest
# token form, the Julia element type, and the KServe V2 wire datatype string. The
# DType -> XLA primitive-type mapping deliberately lives in the Reactant backend,
# not here, so this file has no dependency on Reactant.

using BFloat16s: BFloat16
using DLFP8Types: Float8_E5M2, Float8_E4M3FN

"""
    DType

Canonical element-type enumeration shared across the server, the single source of truth for
dtype translation. The companion maps convert between three representations: the manifest
token form (e.g. `"f32"`, `"bf16"`), the Julia element type (e.g. `Float32`, `BFloat16`), and
the KServe V2 wire datatype string (e.g. `"FP32"`, `"BF16"`).

The `DType` to XLA primitive-type mapping deliberately lives in the Reactant backend, not
here, so this layer carries no Reactant dependency.

FP8 (`F8E5M2`, `F8E4M3`) has no standard KServe wire datatype, so those two variants are
intentionally absent from the wire mapping and may appear only on executable-internal tensors,
never on client-facing inputs or outputs. Conversions are performed by `dtype_from_token`,
`dtype_token`, `julia_type`, `dtype_of`, `dtype_size`, `kserve_string`, and `dtype_from_kserve`.
"""
@enum DType begin
    F16
    F32
    F64
    BF16
    F8E5M2
    F8E4M3
    I8
    I16
    I32
    I64
    U8
    U16
    U32
    U64
    BOOL
end

const DTYPE_FROM_TOKEN = Dict{String,DType}(
    "f16" => F16, "f32" => F32, "f64" => F64, "bf16" => BF16,
    "f8_e5m2" => F8E5M2, "f8_e4m3" => F8E4M3,
    "i8" => I8, "i16" => I16, "i32" => I32, "i64" => I64,
    "u8" => U8, "u16" => U16, "u32" => U32, "u64" => U64,
    "bool" => BOOL,
)
const DTYPE_TO_TOKEN = Dict{DType,String}(v => k for (k, v) in DTYPE_FROM_TOKEN)

const DTYPE_TO_JULIA = Dict{DType,DataType}(
    F16 => Float16, F32 => Float32, F64 => Float64, BF16 => BFloat16,
    F8E5M2 => Float8_E5M2, F8E4M3 => Float8_E4M3FN,
    I8 => Int8, I16 => Int16, I32 => Int32, I64 => Int64,
    U8 => UInt8, U16 => UInt16, U32 => UInt32, U64 => UInt64,
    BOOL => Bool,
)
const JULIA_TO_DTYPE = Dict{DataType,DType}(v => k for (k, v) in DTYPE_TO_JULIA)

# KServe V2 / Open Inference Protocol datatype strings. FP8 has no standard KServe
# enum, so the two f8 tokens are intentionally absent from the wire mapping.
const DTYPE_TO_KSERVE = Dict{DType,String}(
    BOOL => "BOOL",
    U8 => "UINT8", U16 => "UINT16", U32 => "UINT32", U64 => "UINT64",
    I8 => "INT8", I16 => "INT16", I32 => "INT32", I64 => "INT64",
    F16 => "FP16", F32 => "FP32", F64 => "FP64", BF16 => "BF16",
)
const KSERVE_TO_DTYPE = Dict{String,DType}(v => k for (k, v) in DTYPE_TO_KSERVE)

dtype_from_token(s::AbstractString) =
    get(() -> throw(ArgumentError("unknown dtype token: $s")), DTYPE_FROM_TOKEN, String(s))

dtype_token(dt::DType) = DTYPE_TO_TOKEN[dt]

julia_type(dt::DType) = DTYPE_TO_JULIA[dt]

dtype_of(::Type{T}) where {T} =
    get(() -> throw(ArgumentError("no DType for Julia type $T")), JULIA_TO_DTYPE, T)

dtype_size(dt::DType) = sizeof(julia_type(dt))

kserve_string(dt::DType) =
    get(() -> throw(ArgumentError("dtype $dt has no KServe datatype mapping")), DTYPE_TO_KSERVE, dt)

dtype_from_kserve(s::AbstractString) =
    get(() -> throw(ArgumentError("unknown KServe datatype: $s")), KSERVE_TO_DTYPE, String(s))

# ---- Client-facing dtype tables (merged from the SimpleKServe vocabulary) ----
#
# Direct KServe-wire-string <-> Julia-type maps used by the client when building and reading
# wire tensors. These cover only the dtypes with a KServe wire datatype (FP8 has none).

const TritonType =
    Union{UInt8,UInt16,UInt32,UInt64,Int8,Int16,Int32,Int64,Float16,Float32,Float64,Bool}

const KSERVE_OUTPUT_DTYPE_TABLE = Dict{String,Type}(
    "FP16" => Float16, "FP32" => Float32, "FP64" => Float64,
    "INT8" => Int8, "INT16" => Int16, "INT32" => Int32, "INT64" => Int64,
    "UINT8" => UInt8, "UINT16" => UInt16, "UINT32" => UInt32, "UINT64" => UInt64,
    "BOOL" => Bool,
)
const KSERVE_OUTPUT_DTYPE_TABLE_REVERSE =
    Dict(value => key for (key, value) in KSERVE_OUTPUT_DTYPE_TABLE)
