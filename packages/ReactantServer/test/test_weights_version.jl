# Host copies of weights carry the version of the file that was actually mapped, not whatever the
# path names at copy time; a weights file written in place under a live mapping fails the load; and
# a node that cannot back a shared region falls back to a private copy. Reactant-free (no compile),
# bundles written with the fixture helpers.

const _RSC = ReactantServer.ReactantServerCore

function _versioned_bundle(root, name, w::Vector{Float32})
    return write_bundle(
        root, name;
        manifest_yaml = """
        format_version: "2.0"
        name: $name
        executable_inputs:
          - {name: x, dtype: f32, shape: c, dims: {c: $(length(w))}}
        executable_outputs:
          - {name: y, dtype: f32, shape: c, dims: {c: $(length(w))}}
        batching:
          compiled_batch_sizes: [1]
        """,
        mlir_text = """
        module {
          func.func @main(%x: tensor<$(length(w))xf32>, %w: tensor<$(length(w))xf32>) -> tensor<$(length(w))xf32> {
            %0 = stablehlo.multiply %x, %w : tensor<$(length(w))xf32>
            return %0 : tensor<$(length(w))xf32>
          }
        }
        """,
        weights = Dict("w" => w), argument_order = ["w"]
    )
end

_region_path(entry, specs, id) = "/dev/shm/rsw-" * replace(entry.name, r"[^A-Za-z0-9_]" => "_") * "-" *
    string(weights_digest(entry.name, specs; content = weights_file_token(id)); base = 16)

@testset "host weights are labeled with the mapped file's version" begin
    shm_ok = Sys.islinux() && isdir("/dev/shm")
    mktempdir() do root
        old = Float32[1, 2, 3, 4]
        name = "wver_" * string(getpid())
        dir = _versioned_bundle(root, name, old)
        entry = ReactantServer.load_bundle_entry(dir)
        path = joinpath(dir, "weights.safetensors")
        @test entry.weights_id == weights_file_id(path)

        # Deploy new weights atomically, with the same size and mtime, after the bundle loaded.
        tmp = mktempdir()
        _versioned_bundle(tmp, name, Float32[9, 9, 9, 9])
        newfile = joinpath(tmp, name, "weights.safetensors")
        run(`touch -r $path $newfile`)
        mv(newfile, path; force = true)
        @test weights_file_id(path) != entry.weights_id       # the path now names a new version

        # The private copy holds the mapped (old) bytes, and the check passes: the mapping is intact.
        @test ReactantServer.host_materialize(ReactantServer.PrivateWeightStore(), entry, ["w"])[1] == old
        if shm_ok
            store = ReactantServer.SharedWeightStore()
            arrs = ReactantServer.host_materialize(store, entry, ["w"])
            @test arrs[1] == old
            specs = [(Float32, (4,))]
            @test isfile(_region_path(entry, specs, entry.weights_id))            # labeled old
            @test !isfile(_region_path(entry, specs, weights_file_id(path)))      # not new
            arrs = nothing
            ReactantServer.host_release!(store, entry.name)
        end
    end
end

@testset "a weights file written in place under the mapping fails the load" begin
    mktempdir() do root
        name = "winplace_" * string(getpid())
        dir = _versioned_bundle(root, name, Float32[1, 2, 3, 4])
        entry = ReactantServer.load_bundle_entry(dir)
        path = joinpath(dir, "weights.safetensors")
        sleep(0.01)
        open(path, "r+") do io                                  # same inode, new bytes at the end
            seekend(io); skip(io, -4); write(io, Float32(42))
        end
        @test_throws "modified in place" ReactantServer.host_materialize(ReactantServer.PrivateWeightStore(), entry, ["w"])
        if Sys.islinux() && isdir("/dev/shm")
            store = ReactantServer.SharedWeightStore()
            @test_throws "modified in place" ReactantServer.host_materialize(store, entry, ["w"])
            @test isempty(store.attached)                        # nothing published or kept
            san = replace(name, r"[^A-Za-z0-9_]" => "_")
            @test !any(f -> startswith(f, "rsw-" * san * "-") && !endswith(f, ".lock"), readdir("/dev/shm"))
        end
    end
end

@testset "a full /dev/shm falls back to a private copy" begin
    if !(Sys.islinux() && isdir("/dev/shm"))
        @test_skip "shared weight store requires Linux /dev/shm"
    else
        mktempdir() do root
            name = "wfull_" * string(getpid())
            entry = ReactantServer.load_bundle_entry(_versioned_bundle(root, name, Float32[5, 6, 7, 8]))
            store = ReactantServer.SharedWeightStore()
            saved = _RSC._RESERVE_REGION[]
            _RSC._RESERVE_REGION[] = (n, b) -> Cint(28)            # ENOSPC
            try
                arrs = @test_logs (:warn, r"private copy") match_mode = :any ReactantServer.host_materialize(store, entry, ["w"])
                @test arrs[1] == Float32[5, 6, 7, 8]
                @test isempty(store.attached)                    # a private copy: nothing to detach
                ReactantServer.host_release!(store, entry.name)  # and releasing it is a no-op
            finally
                _RSC._RESERVE_REGION[] = saved
            end
        end
    end
end
