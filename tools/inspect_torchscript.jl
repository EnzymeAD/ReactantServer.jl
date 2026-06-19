# TorchScript model inspection helper for the StableHLO conversion effort.
#
# Loads a Triton model's TorchScript artifact and reports what the generic converter cannot tell
# you: the inlined-graph op census (so data-dependent ops like prim::If / prim::Loop /
# aten::nonzero / torchvision::nms / aten::gru / aten::lstm are visible), the module tree (to find
# the dense-core boundary for a `wrap`), the concrete forward output structure, and a faithful
# torch.export probe (the same JIT patches + _JitWrapper the real exporter applies) that reports
# the first failing op or data-dependent guard. It deliberately does NOT load Reactant, so it
# starts fast; the failure that blocks conversion happens in torch.export, before any lowering.
#
# Usage:
#   julia --project=packages/ReactantServerExport/test tools/inspect_torchscript.jl <model_dir> \
#       [--size N] [--no-export] [--code-out PATH]
#
#   <model_dir>   Triton model dir containing config.pbtxt and 1/model.pt
#   --size N      concrete size to substitute for dynamic (-1) input dims (default 512)
#   --no-export   skip the torch.export probe (graph census + forward only)
#   --code-out P  dump the top-level scripted .code to P (default: a temp file)

using PythonCall

# torch must import before anything that could pull Reactant's LLVM; this script never loads
# Reactant, but keep the conventional order so the env behaves identically to the converter.
for m in ("torch", "torch.export", "torchax.export", "torchax.ops.jaten", "numpy")
    try
        pyimport(m)
    catch err
        println("FATAL: cannot import '$m' in this environment.")
        println(sprint(showerror, err)[1:min(end, 400)])
        exit(1)
    end
end

function parse_args(args)
    isempty(args) && error("usage: inspect_torchscript.jl <model_dir> [--size N] [--no-export] [--code-out PATH]")
    model_dir = args[1]
    size = 512
    do_export = true
    code_out = ""
    i = 2
    while i <= length(args)
        a = args[i]
        if a == "--size"
            size = parse(Int, args[i+1]); i += 2
        elseif a == "--no-export"
            do_export = false; i += 1
        elseif a == "--code-out"
            code_out = args[i+1]; i += 2
        else
            error("unknown argument: $a")
        end
    end
    return (; model_dir, size, do_export, code_out)
end

const opts = parse_args(ARGS)

# Everything below runs in Python: parse config.pbtxt, load the model, census the inlined graph,
# run a forward, and probe torch.export with the real TorchScript patches.
pyexec("""
import os, re, tempfile, traceback
from collections import Counter
import torch
from torchax.ops import ops_registry as _ops_registry, jaten as _jaten

_TRITON_TO_TORCH = {
    'TYPE_UINT8': torch.uint8, 'TYPE_UINT16': torch.uint16, 'TYPE_UINT32': torch.uint32, 'TYPE_UINT64': torch.uint64,
    'TYPE_INT8': torch.int8, 'TYPE_INT16': torch.int16, 'TYPE_INT32': torch.int32, 'TYPE_INT64': torch.int64,
    'TYPE_FP16': torch.float16, 'TYPE_FP32': torch.float32, 'TYPE_FP64': torch.float64, 'TYPE_BOOL': torch.bool,
}

def _extract_block(text, keyword):
    # Find `keyword [ ... ]` and return the bracketed body.
    m = re.search(keyword + r'\\s*\\[', text)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    for j in range(i, len(text)):
        if text[j] == '[':
            depth += 1
        elif text[j] == ']':
            depth -= 1
            if depth == 0:
                return text[i+1:j]
    return None

def _extract_objects(block):
    objs, depth, start = [], 0, None
    for j, c in enumerate(block):
        if c == '{':
            if depth == 0:
                start = j + 1
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                objs.append(block[start:j])
    return objs

def parse_inputs(config_text):
    block = _extract_block(config_text, 'input')
    inputs = []
    if block is None:
        return inputs
    for obj in _extract_objects(block):
        name = re.search(r'name\\s*:\\s*"([^"]+)"', obj)
        dt = re.search(r'data_type\\s*:\\s*(TYPE_\\w+)', obj)
        dims = re.search(r'dims\\s*:\\s*\\[([^\\]]*)\\]', obj)
        dim_vals = []
        if dims:
            dim_vals = [int(x) for x in re.findall(r'-?\\d+', dims.group(1))]
        inputs.append((name.group(1) if name else '?', dt.group(1) if dt else '?', dim_vals))
    return inputs

def parse_max_batch(config_text):
    m = re.search(r'max_batch_size\\s*:\\s*(\\d+)', config_text)
    return int(m.group(1)) if m else 1

def census(graph):
    # Count node kinds across the graph, recursing into control-flow sub-blocks.
    counts = Counter()
    def walk(g):
        for n in g.nodes():
            counts[n.kind()] += 1
            for b in n.blocks():
                walk(b)
    walk(graph)
    return counts

def describe_output(o, depth=0):
    pad = '  ' * depth
    if isinstance(o, torch.Tensor):
        return f"{pad}Tensor shape={tuple(o.shape)} dtype={o.dtype}"
    if isinstance(o, (tuple, list)):
        head = f"{pad}{type(o).__name__}[{len(o)}]:"
        return head + "\\n" + "\\n".join(describe_output(x, depth+1) for x in o)
    if isinstance(o, dict):
        head = f"{pad}dict[{len(o)}]:"
        return head + "\\n" + "\\n".join(f"{pad}  {k}:\\n" + describe_output(v, depth+2) for k, v in o.items())
    return f"{pad}{type(o).__name__}: {repr(o)[:80]}"

# --- the real TorchScript export patches (mirror PyTorchExportExt._TORCHSCRIPT_PATCHES_PY) ---
def _fix_jit_parameters(mod):
    for name in list(mod._parameters.keys()):
        p = mod._parameters[name]
        if p is not None and not isinstance(p, torch.nn.Parameter):
            mod._parameters[name] = torch.nn.Parameter(p.detach().clone(), requires_grad=False)
    for child in mod._modules.values():
        if child is not None:
            _fix_jit_parameters(child)

class _JitWrapper(torch.nn.Module):
    def __init__(self, jit_mod):
        super().__init__()
        self.jit_mod = jit_mod
    def forward(self, *args):
        return self.jit_mod(*args)

def report(model_dir, size, do_export, code_out):
    cfg_path = os.path.join(model_dir, 'config.pbtxt')
    pt_path = os.path.join(model_dir, '1', 'model.pt')
    config_text = open(cfg_path).read() if os.path.exists(cfg_path) else ''
    inputs = parse_inputs(config_text)
    maxb = parse_max_batch(config_text)

    print('=' * 78)
    print('MODEL:', os.path.basename(os.path.normpath(model_dir)))
    print('  config:', cfg_path, '   model.pt exists:', os.path.exists(pt_path))
    print('  max_batch_size:', maxb)
    print('  inputs (Triton, batch dim prepended at serve time):')
    for nm, dt, dims in inputs:
        print(f'    {nm}: {dt} dims={dims}')

    model = torch.jit.load(pt_path, map_location='cpu')
    model.eval()

    # --- module tree (top two levels) ---
    print('-' * 78)
    print('MODULE TREE (name : type) -- find the dense-core boundary here:')
    for name, sub in model.named_modules():
        depth = name.count('.')
        if name == '' or depth <= 1:
            label = name if name else '<root>'
            print(f'    {"  " * depth}{label} : {type(sub).__name__}')

    # --- op census of the inlined graph ---
    print('-' * 78)
    print('INLINED-GRAPH OP CENSUS (kind : count), data-dependent ops flagged:')
    flags = ('prim::If', 'prim::Loop', 'aten::nonzero', 'aten::index', 'aten::masked_select',
             'aten::unique', 'torchvision::nms', 'aten::gru', 'aten::lstm', 'aten::rnn',
             'aten::_convolution', 'aten::adaptive_max_pool2d', 'aten::item', 'aten::nonzero_numpy')
    try:
        counts = census(model.inlined_graph)
        for kind, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            mark = '  <-- DATA-DEPENDENT / NEEDS ATTENTION' if any(f in kind for f in flags) else ''
            print(f'    {kind}: {c}{mark}')
    except Exception as e:
        print('    (could not census inlined graph:', repr(e)[:120], ')')

    # --- dump top-level scripted source ---
    out = code_out or tempfile.mkstemp(prefix='ts_code_', suffix='.txt')[1]
    try:
        with open(out, 'w') as f:
            f.write(str(model.code))
        print('-' * 78)
        print('SCRIPTED .code written to:', out)
    except Exception as e:
        print('    (could not dump .code:', repr(e)[:120], ')')

    # --- build example inputs and run a forward ---
    print('-' * 78)
    print('FORWARD on zero inputs (batch=1; dynamic dims -> %d):' % size)
    args = []
    for nm, dt, dims in inputs:
        shape = [1] + [size if d < 0 else d for d in dims]
        args.append(torch.zeros(shape, dtype=_TRITON_TO_TORCH[dt]))
    try:
        with torch.no_grad():
            o = model(*args)
        print(describe_output(o))
    except Exception as e:
        print('    FORWARD FAILED:', repr(e)[:300])

    # --- torch.export probe (faithful: param-fix + handlers + _JitWrapper, strict=False) ---
    if do_export:
        print('-' * 78)
        print('torch.export PROBE (strict=False, with TorchScript patches):')
        try:
            _fix_jit_parameters(model)
            wrapper = _JitWrapper(model)
            ep = torch.export.export(wrapper, tuple(args), strict=False)
            print('    torch.export SUCCEEDED.')
            epc = Counter(str(n.target) for n in ep.graph.nodes() if n.op == 'call_function')
            print('    exported call_function ops (top 25):')
            for tgt, c in sorted(epc.items(), key=lambda kv: (-kv[1], kv[0]))[:25]:
                print(f'        {tgt}: {c}')
        except Exception as e:
            print('    torch.export FAILED with:')
            tb = traceback.format_exc()
            print('   ', repr(e)[:400])
            # surface the data-dependent guard line if present
            for line in tb.splitlines():
                if 'DataDependent' in line or 'guard' in line.lower() or 'specialize' in line.lower():
                    print('    >>', line.strip()[:200])
    print('=' * 78)
""", @__MODULE__)

pyeval("report", @__MODULE__)(opts.model_dir, opts.size, opts.do_export, opts.code_out)
