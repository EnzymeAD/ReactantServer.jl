# Precompile workload: parse a node file and the common manifest forms at package build time, so a
# worker does not compile the YAML reader and the manifest and config parsers while it starts. On a
# CPU worker with four models these cost about 3 s of startup compile before this existed.
#
# The node file uses a production-shaped layout (explicit workers, a global runtime, scheduler and
# endpoints block). The manifests cover the forms bundles use: block and flow style, a batched image
# input, a fixed-shape model, compiled shape variants (`input_shapes`), client-facing I/O that differs
# from the executable's, and a meta model. Files are written to a temporary directory and removed.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    _pc_node(dir) = """
    model_repo: $dir
    base_port: 8080
    metrics_base_port: 9100
    global:
      cache_dir: $dir
      runtime:
        backend: cuda
        autotune: false
        device_memory_fraction: 0.9
        preallocate: true
        allow_cpu_fallback: false
        device_weight_fraction: 1.0
        shared_host_weights: true
        shared_host_weights_mode: "660"
      scheduler:
        discipline: edf
        ema_halflife_seconds: 30.0
        max_queue_depth: 1024
      endpoints:
        host: 0.0.0.0
    workers:
      - { name: worker0 }
      - { name: worker1 }
    """
    _pc_manifests = [
        # A batched image classifier in the exporter's block style, with a client-facing output.
        """
        format_version: "2.1"
        name: "classifier"
        executable_outputs:
          - dtype: "f32"
            name: "logits"
            shape: "an"
            dims:
              a: 9
        provenance:
          source_framework: "pytorch"
          batch_sizes:
            - 1
            - 4
        executable_inputs:
          - dtype: "u8"
            name: "INPUT__0"
            shape: "acn"
            dims:
              c: 224
              a: 224
        batching:
          compiled_batch_sizes:
            - 1
            - 4
        client_outputs:
          - dtype: "i64"
            name: "OUTPUT__0"
            shape: "an"
            dims:
              a: 1
        """,
        # A fixed-shape model in flow style.
        """
        format_version: "2.0"
        name: scale
        executable_inputs:
          - { name: x, dtype: f32, shape: c, dims: { c: 4 } }
        executable_outputs:
          - { name: y, dtype: f32, shape: c, dims: { c: 4 } }
        batching:
          compiled_batch_sizes: [1]
        """,
        # A detector compiled for several input shapes.
        """
        format_version: "2.0"
        name: detector
        executable_inputs:
          - { name: INPUT__0, dtype: f32, shape: whn, dims: { w: -1, h: -1 } }
        executable_outputs:
          - { name: OUTPUT__0, dtype: f32, shape: dcn, dims: { d: 100, c: 6 } }
          - { name: NUM_DETECTIONS, dtype: i64, shape: an, dims: { a: 1 } }
        batching:
          compiled_batch_sizes: [1]
        input_shapes:
          - { w: 1024, h: 1024 }
          - { w: 1448, h: 720 }
        client_outputs:
          - { name: OUTPUT__0, dtype: f32, shape: dc, dims: { d: -1, c: 6 } }
        """,
        # A meta model.
        """
        format_version: "2.0"
        name: pipeline
        kind: meta
        meta:
          calls: [detector, classifier]
        client_inputs:
          - { name: INPUT__0, dtype: u8, shape: whcn, dims: { w: 640, h: 640, c: 3 } }
        client_outputs:
          - { name: OUTPUT__0, dtype: f32, shape: dc, dims: { d: -1, c: 6 } }
        """,
    ]

    # The INFERENCE_SERVER_* overrides unset for the workload, so the build's (or a run-time
    # precompile's) environment cannot change, or invalidate, the config the workload parses.
    _pc_clean_env() = [k => nothing for k in keys(ENV) if startswith(k, ENV_PREFIX)]

    @compile_workload withenv(_pc_clean_env()...) do
        mktempdir() do dir
            path = joinpath(dir, "node.yaml")
            write(path, _pc_node(dir))
            node = load_node(path)
            cfg, _, _ = node_server_config(node, "worker0")
            validate_config(cfg)
            for (i, text) in enumerate(_pc_manifests)
                mpath = joinpath(dir, "manifest$i.yaml")
                write(mpath, text)
                m = load_manifest(mpath)
                client_input_spec(m)
                client_output_spec(m)
            end
        end
    end
end
