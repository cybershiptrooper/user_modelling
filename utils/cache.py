import transformer_lens as tl
import torch
from typing import Callable


def get_cache_fwd_and_bwd(
    model: tl.HookedTransformer,
    tokens: torch.Tensor,
    metric: Callable,
    hook_points: list[str],
):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    for hook_point in hook_points:
        model.add_hook(hook_point, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    for hook_point in hook_points:
        model.add_hook(hook_point, backward_cache_hook, "bwd")
    torch.set_grad_enabled(True)
    logits = model(tokens.clone())
    value = metric(logits)
    value.backward()
    torch.set_grad_enabled(False)
    model.reset_hooks()
    return (
        value.item(),
        tl.ActivationCache(cache, model),
        tl.ActivationCache(grad_cache, model),
    )
