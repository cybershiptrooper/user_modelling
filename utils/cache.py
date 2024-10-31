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
    metric_needs_cache: bool = False,
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
                f"blocks.{layer}.attn.hook_pattern"
                for layer in range(model.cfg.n_layers)
            ]
        else:
            raise ValueError(
                f"Invalid hook_points: {hook_points}, can only be 'all', 'resid_pre', 'resid_post', or 'pattern'"
            )

    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        if not metric_needs_cache:
            cache[hook.name] = act.detach().clone().to(device)
        else:
            cache[hook.name] = act

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
    if metric_needs_cache:
        value = metric(cache)
    else:
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
    hook_point_substring: str = "resid_post",
    padding_side: Literal["left", "right"] = "right",
    pos_slice: slice | None = None,
):
    cache = {}
    prompt_tokenized = model.to_tokens(
        prompts, padding_side=padding_side, prepend_bos=True
    )
    hook_points = [
        hook.name for hook in model.hook_points() if hook_point_substring in hook.name
    ]

    def hook_fn(act, hook):
        if pos_slice is None:
            to_cache = act.detach().to(device)
        else:
            to_cache = (act[pos_slice]).detach().to(device)
        if hook.name in cache:
            cache[hook.name] = torch.cat([cache[hook.name], to_cache], dim=0)
        else:
            cache[hook.name] = to_cache

    caching_hooks = [(hook_point, hook_fn) for hook_point in hook_points]

    for i in tqdm(range(0, len(prompts), batch_size)):
        torch.cuda.empty_cache()
        with model.hooks(fwd_hooks=caching_hooks):
            model(prompt_tokenized[i : i + batch_size], return_type="logits")
        torch.cuda.empty_cache()
    return tl.ActivationCache(cache, model)


def make_mean_cache(
    model: tl.HookedTransformer,
    prompts: list[str],
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    hook_point_substring: str = "resid_post",
    padding_side: Literal["left", "right"] = "right",
    pos_slice: slice | None = None,
):
    cache = {}
    prompt_tokenized = model.to_tokens(
        prompts, padding_side=padding_side, prepend_bos=True
    )
    hook_points = [
        hook.name for hook in model.hook_points() if hook_point_substring in hook.name
    ]

    def hook_fn(act, hook):
        if pos_slice is None:
            to_cache = act.detach().to(device)
        else:
            to_cache = (act[pos_slice]).detach().to(device)
        if hook.name in cache:
            cache[hook.name] += to_cache.sum(dim=0, keepdim=True) / len(prompts)
        else:
            cache[hook.name] = to_cache.sum(dim=0, keepdim=True) / len(prompts)

    caching_hooks = [(hook_point, hook_fn) for hook_point in hook_points]

    for i in tqdm(range(0, len(prompts), batch_size)):
        torch.cuda.empty_cache()
        with model.hooks(fwd_hooks=caching_hooks):
            model(prompt_tokenized[i : i + batch_size], return_type="logits")
        torch.cuda.empty_cache()
    return tl.ActivationCache(cache, model)
