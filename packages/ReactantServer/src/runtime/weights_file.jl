# Reading a bundle's weights.safetensors with pread rather than through a memory mapping.
#
# A memory-mapped file turns a file truncated underneath the reader (`cp new old` truncates the
# destination before writing it) into SIGBUS on the next touch of a page past the new end, which
# kills the worker with no exception to catch. Reading with pread turns the same event into a short
# read, which is an ordinary error and a failed load. Throughput is the same: both go through the
# page cache, and the host copy is a separate array either way (the mapping was only ever the
# source of a copy).
#
# The file stays open from bundle load until the model is unloaded, so an atomic deploy (a new file
# renamed over the old one) keeps reading the version that was loaded, consistent with the compiled
# programs and with `entry.weights_id`, until the watcher reloads the model.
#
# Tensors come back exactly as SafeTensors.jl returns them: a Julia array of the header's `shape`,
# rearranged from the file's row-major order (SafeTensors views the bytes reshaped to the reversed
# shape and permutes the dimensions back; `read_weight` materializes the same array).

ENDIAN_BOM == 0x04030201 || error("ReactantServer reads safetensors (little-endian) only on little-endian hosts")

struct WeightTensorInfo
    eltype::DataType
    shape::Vector{Int}
    offset::Int                 # absolute byte offset of the tensor's data in the file
    nbytes::Int
end

mutable struct WeightsFile
    path::String
    io::IOStream                # kept open (and referenced) so the loaded inode stays readable
    metadata::Union{Dict{String, String}, Nothing}
    tensors::Dict{String, WeightTensorInfo}
end

Base.haskey(w::WeightsFile, name) = haskey(w.tensors, name)
Base.keys(w::WeightsFile) = keys(w.tensors)
Base.length(w::WeightsFile) = length(w.tensors)
# Materializes the tensor (unlike SafeTensors' lazy view); prefer `read_weight` in new code.
Base.getindex(w::WeightsFile, name) = read_weight(w, name)

_fd(w::WeightsFile) = reinterpret(Cint, fd(w.io))

const _ST_MAX_HEADER = 100_000_000          # the bound SafeTensors.jl enforces
const _PREAD_CHUNK = 1 << 26                # 64 MiB per pread call

# Read exactly `nbytes` at `offset` into `ptr`, in chunks. gc_safe: a disk read can take a while, and
# no Julia object crosses the call (the caller roots the destination).
function _pread_full!(w::WeightsFile, ptr::Ptr{UInt8}, nbytes::Int, offset::Int)
    fdc = _fd(w)
    done = 0
    while done < nbytes
        n = min(nbytes - done, _PREAD_CHUNK)
        r = @ccall gc_safe = true pread(fdc::Cint, (ptr + done)::Ptr{Cvoid}, n::Csize_t, Int64(offset + done)::Int64)::Cssize_t
        if r < 0
            err = Libc.errno()
            err == Libc.EINTR && continue
            throw(SystemError("reading weights from $(w.path)", err))
        elseif r == 0
            throw(ErrorException("weights file $(w.path) ended early: it was truncated or replaced in place while loading"))
        end
        done += r
    end
    return nothing
end

"""
    open_weights_file(path) -> (WeightsFile, WeightsFileId)

Open a bundle's `weights.safetensors`, parse and validate its header, and record which on-disk
version was opened (from the open file, so it names the bytes later reads will see). The file is
kept open; close it with [`close_weights!`](@ref) when the model is unloaded.
"""
function open_weights_file(path::AbstractString)
    io = open(String(path), "r")
    try
        id = weights_file_id(stat(io))
        w = WeightsFile(String(path), io, nothing, Dict{String, WeightTensorInfo}())
        fsize = Int(stat(io).size)
        fsize >= 8 || throw(BundleError("$(path): not a safetensors file (shorter than its header)"))
        hdr = Vector{UInt8}(undef, 8)
        GC.@preserve hdr _pread_full!(w, pointer(hdr), 8, 0)
        n = Int(ltoh(reinterpret(UInt64, hdr)[1]))
        (n <= _ST_MAX_HEADER && 8 + n <= fsize) || throw(BundleError("$(path): invalid safetensors header length"))
        json = Vector{UInt8}(undef, n)
        GC.@preserve json _pread_full!(w, pointer(json), n, 8)
        data_start = 8 + n
        for (k, v) in JSON3.read(json)
            key = String(k)
            if key == "__metadata__"
                w.metadata = Dict{String, String}(String(mk) => String(mv) for (mk, mv) in v)
                continue
            end
            tag = Symbol(String(v.dtype))
            (isdefined(SafeTensors, tag) && getfield(SafeTensors, tag) isa SafeTensors.Dtype) ||
                throw(BundleError("$(path): tensor '$key' has unsupported dtype $(tag)"))
            T = SafeTensors.tag2type(getfield(SafeTensors, tag))
            shape = Int[Int(d) for d in v.shape]
            s, e = Int(v.data_offsets[1]), Int(v.data_offsets[2])
            nbytes = e - s
            (s >= 0 && nbytes == prod(shape; init = 1) * sizeof(T) && data_start + e <= fsize) ||
                throw(BundleError("$(path): tensor '$key' has inconsistent offsets or size"))
            w.tensors[key] = WeightTensorInfo(T, shape, data_start + s, nbytes)
        end
        return w, id
    catch
        close(io)
        rethrow()
    end
end

"""
    close_weights!(weights) -> nothing

Close a [`WeightsFile`](@ref) (releasing the inode it holds, which matters once the path was
replaced). A no-op for any other weights handle.
"""
close_weights!(w::WeightsFile) = (isopen(w.io) && close(w.io); nothing)
close_weights!(_) = nothing

"""
    weight_spec(weights, name) -> (eltype, dims)

A tensor's element type and Julia dimensions, without reading its data.
"""
function weight_spec(w::WeightsFile, name)
    info = w.tensors[name]
    return (info.eltype, Tuple(info.shape))
end
weight_spec(st, name) = (eltype(st[name]), size(st[name]))

# Read a tensor's raw (file-order) elements.
function _read_raw(w::WeightsFile, info::WeightTensorInfo)
    raw = Vector{info.eltype}(undef, prod(info.shape; init = 1))
    GC.@preserve raw _pread_full!(w, Ptr{UInt8}(pointer(raw)), info.nbytes, info.offset)
    return raw
end

_perm(N) = ntuple(i -> N + 1 - i, N)

"""
    read_weight(weights, name) -> Array

Read one tensor into a new contiguous Array of its Julia shape (what `collect(st[name])` gives for a
SafeTensors handle).
"""
function read_weight(w::WeightsFile, name)
    info = w.tensors[name]
    raw = _read_raw(w, info)
    N = length(info.shape)
    N <= 1 && return reshape(raw, Tuple(info.shape))
    return permutedims(reshape(raw, Tuple(reverse(info.shape))), _perm(N))
end
read_weight(st, name) = collect(st[name])

"""
    read_weight!(dest, weights, name) -> dest

Read one tensor into the preallocated `dest` (of its Julia shape), e.g. a shared-memory region.
"""
function read_weight!(dest::AbstractArray, w::WeightsFile, name)
    info = w.tensors[name]
    (eltype(dest) == info.eltype && size(dest) == Tuple(info.shape)) ||
        throw(DimensionMismatch("destination does not match tensor '$name'"))
    N = length(info.shape)
    if N <= 1 && dest isa Array
        GC.@preserve dest _pread_full!(w, Ptr{UInt8}(pointer(dest)), info.nbytes, info.offset)
    else
        raw = _read_raw(w, info)
        N <= 1 ? copyto!(dest, raw) : permutedims!(dest, reshape(raw, Tuple(reverse(info.shape))), _perm(N))
    end
    return dest
end
read_weight!(dest, st, name) = copyto!(dest, st[name])
