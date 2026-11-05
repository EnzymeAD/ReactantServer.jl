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

Create the POSIX shared-memory object `key` with `O_CREAT | O_EXCL`, size it to `len` bytes, and map
it read-write through the descriptor, which is kept open. Returns `nothing` when the name already
exists (EEXIST), so the caller can retry under a fresh name; any other failure throws `SystemError`
after unlinking whatever was created. The caller owns the mapping and the descriptor.
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
        identity, _ = _fstat_identity(fd.fd)
        ptr = _shm_mmap(fd.fd, len, key)
        return fd, ptr, identity
    catch
        close(fd)
        ccall(:shm_unlink, Cint, (Cstring,), key)
        rethrow()
    end
end
