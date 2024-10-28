import transformer_lens as tl
import torch
from typing import Callable, Literal
from tqdm import tqdm


def get_cache_fwd_and_bwd(
    model: tl.HookedTransformer,
    x: torch.Tensor | list[str],
    metric: Callable,
    hook_points: list[str] | Literal["all", "resid_pre", "resid_post", "pattern"],
    device: torch.device = torch.device("cpu"),
) -> tuple[float, tl.ActivationCache, tl.ActivationCache]:
    if isinstance(hook_points, str):
        if hook_points == "all":
            return get_cache_for_all_hooks(model, x, metric, device)
        elif hook_points == "resid_pre":
            hook_points = [
                f"blocks.{layer}.hook_resid_pre" for layer in range(model.cfg.n_layers)
            ]
        elif hook_points == "resid_post":
            hook_points = [
                f"blocks.{layer}.hook_resid_post" for layer in range(model.cfg.n_layers)
            ]
        elif hook_points == "pattern":
            hook_points = [
                f"blocks.{layer}.attn.hook_pattern" for layer in range(model.cfg.n_layers)
            ]
        else:
            raise ValueError(
                f"Invalid hook_points: {hook_points}, can only be 'all', 'resid_pre', 'resid_post', or 'pattern'"
            )

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


def get_cache_for_all_hooks(
    model: tl.HookedTransformer,
    x: torch.Tensor | list[str],
    metric: Callable,
    device: torch.device = torch.device("cpu"),
):
    raise NotImplementedError("Not implemented")
    fwd_cache = {}
    bwd_cache = {}

    def fwd_cache_hook():
        def hook(act, hook):
            fwd_cache[hook.name] = act.detach().clone().to(device)

        return (lambda _: True, hook)

    def bwd_cache_hook():
        def hook(act, hook):
            bwd_cache[hook.name] = act.detach().clone().to(device)

        return (lambda _: True, hook)

    with model.hooks(fwd_hooks=[fwd_cache_hook()], bwd_hooks=[bwd_cache_hook()]):
        if isinstance(x, torch.Tensor):
            logits = model(x.clone())
        else:
            logits = model(x, padding_side="left")
        value = metric(logits)
    return (
        value.item(),
        tl.ActivationCache(fwd_cache, model),
        tl.ActivationCache(bwd_cache, model),
    )


def batched_fwd_cache(
    model: tl.HookedTransformer,
    prompts: list[str],
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
):
    cache = {}
    prompt_tokenized = model.tokenizer(
        prompts, padding=True, truncation=True, return_tensors="pt", padding_side="left"
    )["input_ids"]

    for i in tqdm(range(0, len(prompts), batch_size)):
        torch.cuda.empty_cache()
        output, cache_batch = model.run_with_cache(
            prompt_tokenized[i : i + batch_size], return_type="logits"
        )
        cache_batch = cache_batch.to(device)
        for k, v in cache_batch.items():
            if "resid" in k:
                if k not in cache:
                    cache[k] = v.detach().clone()
                else:
                    cache[k] = torch.cat([cache[k], v], dim=0)
    return cache
