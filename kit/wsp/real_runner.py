"""Real-model runner: any open-weight causal LM, via HuggingFace Auto.

Loads with ``AutoModelForCausalLM`` / ``AutoTokenizer`` and discovers
MLP output projections with ``wsp.layers`` — no hardcoded family path.
Two jobs live here and nowhere else:

1. Generate text (greedy or sampled, fixed seeds) and score it.
2. Capture MLP signals during generation (MLP output vectors and
   down-projection inputs at each predicted position), so the pipeline
   can estimate a direction, prove it causal, and edit columns.

Edits are applied to the in-memory model only, on a copy of the
affected columns. Original files are never touched; the runner can
restore the saved columns exactly (bitwise copy-back), and edited
weights are saved to a new directory with ``save_pretrained``.

Requires the ``real`` extra: ``pip install 'wsp-kit[real]'`` (torch,
transformers, accelerate). Importing this module without them raises
a plain ImportError with that hint.
"""

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "real-model support needs torch + transformers: "
        "pip install 'wsp-kit[real]'"
    ) from exc

from . import layers

MAX_PARAMS = 3_000_000_000


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


class RealModel:
    """A loaded causal LM plus the hooks an edit needs."""

    def __init__(self, model_id, dtype="bfloat16", device="cpu",
                 use_chat_template=None, thinking=False):
        self.model_id = model_id
        self.device = device
        self.thinking = thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = getattr(torch, dtype)
        # transformers 5.x renamed torch_dtype -> dtype; older versions
        # only know torch_dtype. Try the new name first, fall back.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, low_cpu_mem_usage=True,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
            )
        self.model.to(device)
        self.model.eval()
        total = count_params(self.model)
        if total > MAX_PARAMS:
            raise ValueError(
                f"model has {total:,} parameters, over the {MAX_PARAMS:,} "
                "cap — too large for weight-space editing with current tooling"
            )
        self.total_params = total
        if use_chat_template is None:
            use_chat_template = self.tokenizer.chat_template is not None
        self.use_chat_template = bool(use_chat_template)
        blocks, _path = layers.find_decoder_blocks(self.model)
        self.blocks = blocks
        self.layer_infos = layers.describe_layers(self.model)
        self._infos_by_index = {info["index"]: info
                                for info in self.layer_infos}
        self._down_projs = {}
        for info in self.layer_infos:
            block = blocks[info["index"]]
            mlp = getattr(block, info["mlp_name"])
            self._down_projs[info["index"]] = getattr(mlp, info["down_proj_name"])
        self._mlps = {}
        for info in self.layer_infos:
            block = blocks[info["index"]]
            self._mlps[info["index"]] = getattr(block, info["mlp_name"])

    # -- prompts -----------------------------------------------------

    def render(self, prompt):
        """Chat template if enabled, else the raw prompt."""
        if self.use_chat_template and self.tokenizer.chat_template:
            tmpl = self.tokenizer.chat_template
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            if isinstance(tmpl, str) and "enable_thinking" in tmpl:
                kwargs["enable_thinking"] = self.thinking
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], **kwargs
            )
        return prompt

    def _eos_id(self):
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = getattr(self.model.generation_config, "eos_token_id", None)
        return eos

    def _pad_id(self):
        pad = self.tokenizer.pad_token_id
        return pad if pad is not None else self._eos_id()

    # -- generation --------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature, seed):
        """Generate once. Returns (text, finished, new_token_count)."""
        text = self.render(prompt)
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self._pad_id(),
            "eos_token_id": self._eos_id(),
        }
        if temperature == 0:
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = float(temperature)
            # transformers 5.x generate() takes no `generator` kwarg
            # (it errors on unused model_kwargs), so seed the global
            # RNG instead. Generation here is strictly sequential —
            # one prompt at a time, no batching, no threads — so a
            # global seed is exactly as reproducible as a per-call
            # generator.
            torch.manual_seed(int(seed))
        out = self.model.generate(**ids, **kwargs)
        new_ids = out[0][ids["input_ids"].shape[1]:]
        finished = bool(len(new_ids) > 0 and int(new_ids[-1]) == self._eos_id())
        return (
            self.tokenizer.decode(new_ids, skip_special_tokens=True),
            finished,
            int(len(new_ids)),
        )

    # -- diagnosis capture -------------------------------------------

    @torch.no_grad()
    def capture(self, prompt, max_new_tokens, layer_indices):
        """Greedy-generate while capturing MLP signals.

        For each requested layer, returns the mean (over generated
        positions) MLP-output vector and mean down-projection-input
        vector, both as float32 numpy arrays. Only the last position
        of each forward pass is captured — the position predicting the
        next token — so memory stays at one vector per layer no matter
        how long the generation runs.
        """
        sums = {i: {"mlp_out": None, "down_in": None, "count": 0}
                for i in layer_indices}

        def mlp_hook(idx):
            def hook(_module, _inp, out):
                vec = out[0, -1, :].detach().to(torch.float32).cpu()
                cur = sums[idx]["mlp_out"]
                sums[idx]["mlp_out"] = vec if cur is None else cur + vec
            return hook

        def down_hook(idx):
            def hook(_module, inp, _out):
                vec = inp[0][0, -1, :].detach().to(torch.float32).cpu()
                cur = sums[idx]["down_in"]
                sums[idx]["down_in"] = vec if cur is None else cur + vec
                sums[idx]["count"] += 1
            return hook

        handles = []
        try:
            for i in layer_indices:
                handles.append(self._mlps[i].register_forward_hook(mlp_hook(i)))
                handles.append(self._down_projs[i].register_forward_hook(down_hook(i)))
            self.generate(prompt, max_new_tokens, 0, 0)
        finally:
            for h in handles:
                h.remove()
        result = {}
        for i in layer_indices:
            count = sums[i]["count"] or 1
            result[i] = {
                "mlp_out": sums[i]["mlp_out"].numpy() / count,
                "down_in": sums[i]["down_in"].numpy() / count,
            }
        return result

    # -- edits ---------------------------------------------------------

    def apply_edit(self, layer_index, columns, direction, alpha):
        """Subtract alpha*direction from columns of one down-projection.

        Returns (changed_count, saved) where saved restores the edit
        exactly. Raises ValueError if the edit breaks the 1% cap.
        """
        from .real_edit import MAX_EDIT_FRACTION, apply_column_edit

        proj = self._down_projs[layer_index]
        info = self._infos_by_index[layer_index]
        layout = info["weight_layout"]
        intended = len(columns) * int(info["hidden_size"])
        frac = intended / self.total_params
        if frac > MAX_EDIT_FRACTION:
            raise ValueError(
                f"edit touches {frac:.2%} of weights, over the 1% cap — "
                "rejected as too broad"
            )
        weight = proj.weight.data
        if layout == "out_in":
            saved = weight[:, list(columns)].clone()
        else:
            saved = weight[list(columns), :].clone()
        new, changed = apply_column_edit(weight, columns, direction, alpha,
                                         layout=layout)
        with torch.no_grad():
            weight.copy_(new)
        return changed, (layer_index, list(columns), saved)

    def restore(self, saved):
        """Copy saved columns back exactly (bitwise restore).

        NOTE: this must use indexed assignment (``=``), not ``.copy_()``
        on an indexed slice — advanced indexing returns a copy, so
        ``weight[:, cols].copy_(orig)`` writes into a temporary and
        leaves the model untouched (a real bug caught by tests).
        """
        layer_index, columns, orig = saved
        proj = self._down_projs[layer_index]
        layout = self._infos_by_index[layer_index]["weight_layout"]
        with torch.no_grad():
            if layout == "out_in":
                proj.weight.data[:, list(columns)] = orig
            else:
                proj.weight.data[list(columns), :] = orig

    def zero_columns(self, layer_index, columns):
        """Zero columns (causal ablation). Returns saved for restore."""
        proj = self._down_projs[layer_index]
        layout = self._infos_by_index[layer_index]["weight_layout"]
        if layout == "out_in":
            saved = proj.weight.data[:, list(columns)].clone()
            with torch.no_grad():
                proj.weight.data[:, list(columns)] = 0
        else:
            saved = proj.weight.data[list(columns), :].clone()
            with torch.no_grad():
                proj.weight.data[list(columns), :] = 0
        return (layer_index, list(columns), saved)

    def save(self, out_dir):
        """Save the (possibly edited) model + tokenizer to a new dir."""
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        return out_dir
