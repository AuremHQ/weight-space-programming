"""Instrumentation: count what a run spends (W2.6).

``CountingModel`` proxies a model, counts generate calls and
generated tokens, and logs (phase, prompt) per call/capture
through ``with proxy.phase("diagnose")``. An "eval pass" is one
generate call (the same meaning as ``merge.count_eval_passes``).
A capture is one greedy generation internally, so it counts as
one pass too (its tokens are not visible — the capture returns
vectors, not token counts — so generated_tokens only sums direct
generate calls).
"""

import contextlib


class CountingModel:
    """A counting proxy around a RealModel-like object."""

    def __init__(self, model):
        self._model = model
        self.eval_passes_spent = 0
        self.generated_tokens = 0
        self.calls = []  # [(phase, prompt, kind)]
        self.phase_prompt_counts = {}  # {phase: {prompt: calls}}
        self._phase = "unphased"

    def __getattr__(self, name):
        return getattr(self._model, name)

    @contextlib.contextmanager
    def phase(self, name):
        """Log the following calls under ``name``."""
        previous = self._phase
        self._phase = name
        try:
            yield self
        finally:
            self._phase = previous

    def _log(self, prompt, kind):
        self.calls.append((self._phase, prompt, kind))
        per_prompt = self.phase_prompt_counts.setdefault(self._phase,
                                                         {})
        per_prompt[prompt] = per_prompt.get(prompt, 0) + 1

    def generate(self, prompt, max_new_tokens, temperature, seed,
                 **kwargs):
        """Count one eval pass; add its generated tokens."""
        text, finished, n_tokens = self._model.generate(
            prompt, max_new_tokens, temperature, seed, **kwargs)
        self.eval_passes_spent += 1
        self.generated_tokens += int(n_tokens)
        self._log(prompt, "generate")
        return text, finished, n_tokens

    def capture(self, prompt, max_new_tokens, layer_indices):
        """Count one eval pass (one greedy generation inside)."""
        result = self._model.capture(prompt, max_new_tokens,
                                     layer_indices)
        self.eval_passes_spent += 1
        self._log(prompt, "capture")
        return result
