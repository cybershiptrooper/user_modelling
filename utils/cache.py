import transformer_lens as tl
import torch
from typing import Callable


def get_cache_fwd_and_bwd(
    model: tl.HookedTransformer,
    x: torch.Tensor | list[str],
    metric: Callable,
    hook_points: list[str],
    device: torch.device = torch.device("cpu"),
) -> tuple[float, tl.ActivationCache, tl.ActivationCache]:
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach().clone().to(device)

    for hook_point in hook_points:
        model.add_hook(hook_point, forward_cache_hook, "fwd")

    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach().clone().to(device)

    for hook_point in hook_points:
        model.add_hook(hook_point, backward_cache_hook, "bwd")
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    if isinstance(x, torch.Tensor):
        logits = model(x.clone())
    else:
        logits = model(x, padding_side="left")
    value = metric(logits)
    value.backward()
    torch.set_grad_enabled(False)
    model.reset_hooks()
    torch.cuda.empty_cache()
    return (
        value.item(),
        tl.ActivationCache(cache, model),
        tl.ActivationCache(grad_cache, model),
    )
