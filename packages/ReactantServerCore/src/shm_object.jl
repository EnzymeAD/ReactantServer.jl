# POSIX shared-memory objects opened, identified and mapped through one retained file descriptor.
#
# InterProcessCommunication's `SharedMemory` opens a name, maps it and closes the descriptor, so the
# object it mapped can no longer be identified afterwards: a second open by name may already reach a
# different object if the name was unlinked and recreated in between. Both ends of the system
# shared-memory transport need more than that. The worker records the (device, inode) of the object
# it maps, taken from the same descriptor it maps, so a re-registration can tell "the same object"
# from "a recreated name" (CORE-2); and both ends keep the descriptor, because the slot control
# protocol takes open-file-description locks on it (CORE-3).

import InterProcessCommunication as IPC

"""
    ShmIdentity(device, inode)

The identity of a POSIX shared-memory object: the (device, inode) pair `fstat` reports for it. Two
opens reach the same object exactly when their identities are equal; a name that was unlinked and
created again resolves to a new inode, and an inode is never reused while a mapping of the old one
is alive.
"""
struct ShmIdentity
    device::UInt64
    inode::UInt64
end

# A retained descriptor of a shared-memory object. Closed explicitly (idempotent) or by the GC.
mutable struct ShmFd
    fd::Cint
    function ShmFd(fd::Integer)
        return finalizer(_close_fd!, new(Cint(fd)))
    end
end

function _close_fd!(f::ShmFd)
    fd = f.fd
    fd < 0 && return nothing
    f.fd = Cint(-1)
    ccall(:close, Cint, (Cint,), fd)
    return nothing
end
Base.close(f::ShmFd) = _close_fd!(f)
Base.isopen(f::ShmFd) = f.fd >= 0

# shm_open(2); returns the descriptor or -errno.
function _shm_open(key::AbstractString, flags::Integer, mode::Integer)
    fd = ccall(:shm_open, Cint, (Cstring, Cint, IPC.mode_t), key, flags, mode)
    return fd < 0 ? -Cint(Libc.errno()) : fd
end

function _fstat_identity(fd::Cint)
    st = stat(RawFD(fd))
    return ShmIdentity(UInt64(st.device), UInt64(st.inode)), Int(st.size)
end

"""
    shm_key_identity(key) -> Union{ShmIdentity, Symbol}

What the POSIX shared-memory name `key` resolves to right now: its [`ShmIdentity`](@ref), `:gone`
when no such object exists (it was unlinked), or `:unknown` when the open failed for any other
reason (a permission change, descriptor exhaustion), which a caller must not read as either.
"""
function shm_key_identity(key::AbstractString)
    fd = _shm_open(key, IPC.O_RDONLY, 0)
    fd == -Libc.ENOENT && return :gone
    fd < 0 && return :unknown
    try
        return first(_fstat_identity(fd))
    catch
        return :unknown
    finally
        ccall(:close, Cint, (Cint,), fd)
    end
end

# mmap(2) `len` bytes of `fd` shared and read-write; throws on failure.
function _shm_mmap(fd::Cint, len::Integer, key::AbstractString)
    ptr = ccall(
        :mmap, Ptr{Cvoid}, (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, IPC.off_t),
        C_NULL, len, IPC.PROT_READ | IPC.PROT_WRITE, IPC.MAP_SHARED, fd, 0
    )
    ptr == IPC.MAP_FAILED && throw(SystemError("mmap of shared memory object '$key'", Libc.errno()))
    return convert(Ptr{UInt8}, ptr)
end

_shm_munmap(ptr::Ptr, len::Integer) = (ccall(:munmap, Cint, (Ptr{Cvoid}, Csize_t), ptr, len); nothing)

# ---- backing pages (CORE-4) ----
#
# A POSIX shared-memory object lives on tmpfs (`/dev/shm`), which allocates a page when it is first
# touched, not when the object is sized. Touching a page through a mapping raises SIGBUS, which kills
# the process, when tmpfs cannot allocate it (`/dev/shm` is full, or the memory limit refuses it) and
# when it lies past the object's end (the object was shrunk after it was mapped). A read fault
# allocates too, so reading a page nobody wrote is no safer than writing it. Both ends therefore make
# sure of the pages before touching them through a mapping: a client pool allocates its whole object
# when it is created, and a server checks a registered region's size and allocates the range before
# every copy into or out of it (shared_memory.jl). `posix_fallocate` reports a page it cannot
# allocate as an error number instead of a signal.

"""
    ShmSpaceError(name, bytes, errno)

The pages of `bytes` bytes of the shared-memory object or region `name` could not be allocated:
`/dev/shm` is full (`ENOSPC`) or the memory limit refused them (`ENOMEM`). Raised where touching an
unallocated page would otherwise kill the process with SIGBUS: when a client's `BufferPool` object is
created (nothing is left behind), and before a server copies into or out of a registered region (the
transport maps it to RESOURCE_EXHAUSTED).
"""
struct ShmSpaceError <: Exception
    name::String
    bytes::Int
    errno::Cint
end
Base.showerror(io::IO, e::ShmSpaceError) = print(
    io, "cannot allocate ", Base.format_bytes(e.bytes), " of shared memory '", e.name,
    "' (/dev/shm is full or the memory limit is reached): ", Libc.strerror(e.errno)
)

const _EDQUOT = Cint(122)   # Linux; tmpfs reports it when a quota refuses the pages

# Allocate every page of [offset, offset + len) of `fd`'s object that has none, keeping every byte
# that is there (a page tmpfs allocates reads as zeros, as an unallocated one did). Returns 0 or an
# error number: posix_fallocate returns it rather than setting errno, so the call can be gc_safe and
# a collection need not wait out a large allocation. It would also extend an object shorter than
# the range; a server checks the size first (shared_memory.jl). A signal can interrupt a large
# allocation (tmpfs undoes it and reports EINTR), which is retried. Linux only; elsewhere it reports
# 0 and nothing is allocated.
function _shm_fallocate(fd::Cint, offset::Integer, len::Integer)
    (Sys.islinux() && len > 0) || return Cint(0)
    while true
        rc = @ccall gc_safe = true posix_fallocate(fd::Cint, Int64(offset)::Int64, Int64(len)::Int64)::Cint
        rc == Libc.EINTR || return rc
    end
    return
end

# The allocation every caller goes through, `(fd, offset, len) -> errno`. A Ref so tests can simulate
# a full /dev/shm; called with `invokelatest`, so a replacement defined after a long-running task
# started (the worker's dispatch loop) is seen by it.
const _SHM_FALLOCATE = Ref{Function}(_shm_fallocate)

# Allocate the pages of [offset, offset + len) of `fd`'s object (named `name` in errors), or throw:
# `ShmSpaceError` when there is no room for them, `SystemError` for any other failure.
function _shm_back!(fd::Cint, offset::Integer, len::Integer, name::AbstractString)
    rc = Cint(Base.invokelatest(_SHM_FALLOCATE[], fd, offset, len))
    rc == 0 && return nothing
    (rc == Libc.ENOSPC || rc == Libc.ENOMEM || rc == _EDQUOT) &&
        throw(ShmSpaceError(String(name), Int(len), rc))
    throw(SystemError("posix_fallocate of shared memory '$name'", rc))
end

# The current size of `fd`'s object, and whether one fstat proves every page of it allocated, so that
# a touch anywhere inside it needs no new page. tmpfs counts an object's allocated pages, swapped ones
# included, in st_blocks (512-byte units); blocks equal to the size rounded up to whole pages leave
# no room for a hole. Only pages past the end can hide one, and tmpfs keeps those only when a client
# allocates there on purpose (fallocate with FALLOC_FL_KEEP_SIZE) or in a huge page straddling the
# end; even then they must match the holes exactly. A dense object in huge pages has more blocks
# than that and is not proven dense, which costs the caller speed, not safety. Linux only; elsewhere
# it never proves density.
function _fstat_size_dense(fd::ShmFd)
    st = stat(RawFD(fd.fd))
    size = Int(st.size)
    Sys.islinux() || return size, false
    page = _page_size()
    return size, Int(st.blocks) * 512 == cld(size, page) * page
end

"""
    shm_open_existing(key) -> (fd::ShmFd, ptr::Ptr{UInt8}, len::Int, identity::ShmIdentity)

Open the existing POSIX shared-memory object `key` read-write, `fstat` the descriptor, and map the
whole object through that same descriptor, which is kept open. The identity therefore names exactly
the object that is mapped. Throws `ArgumentError` when the object does not exist or is empty and
`SystemError` for any other failure; nothing is left open on a throw. The caller owns the mapping
(`munmap` it with its length) and the descriptor.
"""
function shm_open_existing(key::AbstractString)
    raw = _shm_open(key, IPC.O_RDWR, 0)
    raw == -Libc.ENOENT && throw(ArgumentError("shared memory object '$key' does not exist"))
    raw < 0 && throw(SystemError("shm_open of shared memory object '$key'", -raw))
    fd = ShmFd(raw)
    try
        identity, len = _fstat_identity(fd.fd)
        len > 0 || throw(ArgumentError("shared memory object '$key' is empty"))
        ptr = _shm_mmap(fd.fd, len, key)
        return fd, ptr, len, identity
    catch
        close(fd)
        rethrow()
    end
end

"""
    shm_create_exclusive(key, len; mode = 0o600) -> Union{Nothing, Tuple{ShmFd, Ptr{UInt8}, ShmIdentity}}

Create the POSIX shared-memory object `key` with `O_CREAT | O_EXCL`, size it to `len` bytes, allocate
all of its pages, and map it read-write through the descriptor, which is kept open. Returns `nothing`
when the name already exists (EEXIST), so the caller can retry under a fresh name. When `/dev/shm`
cannot hold `len` bytes it throws [`ShmSpaceError`](@ref), and any other failure throws
`SystemError`, in both cases after unlinking whatever was created. The caller owns the mapping and
the descriptor.

Sizing an object allocates none of its pages, so without the allocation a full `/dev/shm` let the
object be created and mapped, and the first write past the space left killed the process with
SIGBUS (CORE-4). Allocating up front commits the object's whole size to `/dev/shm` at once.
"""
function shm_create_exclusive(key::AbstractString, len::Integer; mode::Integer = 0o600)
    len > 0 || throw(ArgumentError("shared memory object size must be positive, got $len"))
    raw = _shm_open(key, IPC.O_CREAT | IPC.O_EXCL | IPC.O_RDWR, mode | 0o600)
    raw == -Libc.EEXIST && return nothing
    raw < 0 && throw(SystemError("shm_open (create) of shared memory object '$key'", -raw))
    fd = ShmFd(raw)
    try
        if ccall(:ftruncate, Cint, (Cint, IPC.off_t), fd.fd, len) != 0
            throw(SystemError("ftruncate of shared memory object '$key'", Libc.errno()))
        end
        _shm_back!(fd.fd, 0, len, key)
        identity, _ = _fstat_identity(fd.fd)
        ptr = _shm_mmap(fd.fd, len, key)
        return fd, ptr, identity
    catch
        close(fd)
        ccall(:shm_unlink, Cint, (Cstring,), key)
        rethrow()
    end
end
