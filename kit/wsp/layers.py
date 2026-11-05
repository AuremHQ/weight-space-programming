"""Find decoder blocks and MLP output projections, generically.

Model families name things differently — Qwen/Llama/Mistral/Gemma call
it ``model.layers[i].mlp.down_proj``, GPT-2 calls it
``transformer.h[i].mlp.c_proj``, Falcon calls it
``transformer.h[i].mlp.dense_4h_to_h`` — but the shape is the same: a
stack of decoder blocks, each with an MLP whose last linear map sends
the intermediate activations back into the residual stream. That last
map is what a weight-space edit touches (its columns are vectors in
residual space), so this module finds it without hardcoding one
family's module path.

GPT-2's ``c_proj`` is a Conv1D, not an nn.Linear: it carries ``nf``
/``nx`` instead of ``out_features``/``in_features``, and its weight
is stored transposed — shape (intermediate, hidden) instead of
(hidden, intermediate). The finder records that as ``weight_layout``
("in_out" vs the usual "out_in") so the edit code can touch the
right axis.

Nothing here imports torch: the finder walks attributes and checks
for linear-like objects (``in_features``/``out_features`` or
``nf``/``nx``, plus a 2-D ``weight``). It works on real models and
on stub objects in tests.
"""

# Attribute chains from the model object to its list of decoder blocks,
# in the order to try them. First chain that yields a non-empty list wins.
BLOCK_PATHS = (
    ("model", "layers"),          # Qwen, Llama, Mistral, Gemma, Phi-3, SmolLM2
    ("transformer", "h"),         # GPT-2, Falcon
    ("gpt_neox", "layers"),       # GPT-NeoX
    ("model", "decoder", "layers"),  # OPT, BART decoder
    ("decoder", "layers"),        # OPT (alternate root)
    ("layers",),                  # models that expose blocks directly
    ("transformer", "blocks"),    # misc decoder stacks
)

# MLP submodule names inside a decoder block, in the order to try.
MLP_NAMES = ("mlp", "feed_forward")

# Output-projection names inside an MLP, in the order to try. The first
# linear-like match wins. NOTE: attention output projections are never
# considered — only names inside the MLP submodule.
DOWN_PROJ_NAMES = ("down_proj", "c_proj", "dense_4h_to_h", "fc_out", "fc2")


def _walk(obj, path):
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def _is_list_like(obj):
    try:
        return len(obj) > 0 and obj[0] is not None
    except (TypeError, IndexError, KeyError):
        return False


def _linear_dims(obj):
    """(in_features, out_features, layout) for a linear-like module.

    nn.Linear layout ("out_in"): weight shape (out, in), with
    ``in_features``/``out_features`` attributes. Conv1D layout
    ("in_out", GPT-2's ``c_proj``): weight shape (in, out), with
    ``nf`` (out) / ``nx`` (in) attributes. Returns None when the
    object is not linear-like in either layout.
    """
    if obj is None:
        return None
    weight = getattr(obj, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    in_f = getattr(obj, "in_features", None)
    out_f = getattr(obj, "out_features", None)
    if isinstance(in_f, int) and isinstance(out_f, int):
        if tuple(shape) == (out_f, in_f):
            return in_f, out_f, "out_in"
        return None
    nf = getattr(obj, "nf", None)
    nx = getattr(obj, "nx", None)
    if isinstance(nf, int) and isinstance(nx, int):
        if tuple(shape) == (nx, nf):
            return nx, nf, "in_out"
    return None


def _is_linear_like(obj):
    return _linear_dims(obj) is not None


def find_decoder_blocks(model):
    """Return (blocks, path_used). Raises ValueError if no stack found."""
    for path in BLOCK_PATHS:
        blocks = _walk(model, path)
        if _is_list_like(blocks):
            return list(blocks), ".".join(path)
    raise ValueError(
        "could not find a decoder block list — tried "
        + ", ".join(".".join(p) for p in BLOCK_PATHS)
        + ". If this model family is new, add its path to BLOCK_PATHS."
    )


def find_mlp_down_proj(block):
    """Return (mlp, down_proj, name) for one block, or None."""
    mlp = None
    for name in MLP_NAMES:
        mlp = getattr(block, name, None)
        if mlp is not None:
            break
    if mlp is None:
        return None
    for name in DOWN_PROJ_NAMES:
        proj = getattr(mlp, name, None)
        if _is_linear_like(proj):
            return mlp, proj, name
    return None


def describe_layers(model):
    """One LayerInfo dict per decoder block with a findable MLP output.

    LayerInfo fields: index, block_path, mlp_name, down_proj_name,
    weight_layout ("out_in" for nn.Linear, "in_out" for Conv1D —
    which axis holds the intermediate columns), hidden_size
    (down_proj.out_features / Conv1D nf), intermediate_size
    (down_proj.in_features / Conv1D nx), down_proj_numel. Blocks
    without a findable MLP output are skipped, not faked.
    """
    blocks, path_used = find_decoder_blocks(model)
    infos = []
    for i, block in enumerate(blocks):
        found = find_mlp_down_proj(block)
        if found is None:
            continue
        mlp, proj, name = found
        dims = _linear_dims(proj)
        mlp_name = next(n for n in MLP_NAMES if getattr(block, n, None) is mlp)
        in_f, out_f, layout = dims
        infos.append({
            "index": i,
            "block_path": f"{path_used}[{i}]",
            "mlp_name": mlp_name,
            "down_proj_name": name,
            "weight_layout": layout,
            "hidden_size": int(out_f),
            "intermediate_size": int(in_f),
            "down_proj_numel": int(proj.weight.numel()),
        })
    if not infos:
        raise ValueError(
            "found decoder blocks but no MLP output projection in any of "
            f"them (tried {list(DOWN_PROJ_NAMES)} inside {list(MLP_NAMES)})"
        )
    return infos
