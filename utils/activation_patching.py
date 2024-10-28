import torch
import transformer_lens as tl
from transformer_lens.hook_points import HookPoint
from utils.index import TorchIndex, Ix
from typing import Callable
from tqdm import tqdm


def make_ablation_hook(
    cache: dict[str, torch.Tensor] | tl.ActivationCache,
    index: TorchIndex = Ix[[None]],
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    def ablation_hook(hook_point_out: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        out = hook_point_out.clone()
        out[index.as_index] = cache[hook.name][index.as_index]
        return out

    return ablation_hook


def patch_resid_at_each_token_pos(
    model: tl.HookedTransformer,
    clean_prompt: str,
    counterfactual_prompt: str,
    corrupted_cache: dict,
    metric: Callable,
    use_resid_post: bool = False,
) -> torch.Tensor:
    """
    Run node-wise patching across all layers and positions.

    Args:
        model: The transformer model
        clean_prompt: The clean input prompt
        counterfactual_prompt: The corrupted input prompt
        corrupted_cache: Cache from running the corrupted input
        resid_hook_points: List of residual hook points to patch
        metric: Metric function to evaluate outputs

    Returns:
        Tensor of shape (n_layers, seq_len) containing patching results
    """
    seq_len = len(model.to_str_tokens(counterfactual_prompt))
    n_layers = model.cfg.n_layers
    results = torch.zeros(n_layers, seq_len)
    resid_hook_points = [
        (
            f"blocks.{i}.hook_resid_pre"
            if not use_resid_post
            else f"blocks.{i}.hook_resid_post"
        )
        for i in range(n_layers)
    ]

    for layer in tqdm(range(n_layers)):
        for pos in range(seq_len):
            torch.cuda.empty_cache()
            hook_point = resid_hook_points[layer]
            out = model.run_with_hooks(
                clean_prompt,
                fwd_hooks=[
                    (
                        hook_point,
                        make_ablation_hook(cache=corrupted_cache, index=Ix[:, pos]),
                    )
                ],
            )
            results[layer, pos] = metric(out).item()
            torch.cuda.empty_cache()

    return results
