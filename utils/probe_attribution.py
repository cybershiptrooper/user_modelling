from transformer_lens import ActivationCache
import torch
import transformer_lens as tl
from utils.attribution import compute_attribution_patching_scores_for_hook_points

def compute_model_attribution_patching_scores_with_mean_cache(
    model: tl.HookedTransformer,
    clean_fwd_cache: tl.ActivationCache,
    clean_bwd_cache: tl.ActivationCache,
    mean_cache: dict | tl.ActivationCache,
    hook_points: list[str],
    device: str = "cpu",
):
    corrupt_cache = make_corrupt_cache_with_correct_shape(
        mean_cache, hook_points, clean_fwd_cache, model
    )
    corrupt_cache = corrupt_cache.to(device)
    clean_fwd_cache = clean_fwd_cache.to(device)
    clean_bwd_cache = clean_bwd_cache.to(device)

    attrs = compute_attribution_patching_scores_for_hook_points(
        clean_fwd_cache,
        corrupt_cache,
        clean_bwd_cache,
        hook_points,
        zero_ablation=False,
        use_corrupt_bwd_cache=False,
    )

    del corrupt_cache, clean_fwd_cache, clean_bwd_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return attrs


def make_corrupt_cache_with_correct_shape(
    mean_cache: dict | tl.ActivationCache, 
    hook_points: list[str], 
    clean_fwd_cache: tl.ActivationCache, 
    model: tl.HookedTransformer
):
    corrupt_cache = {}
    for hook_point in hook_points:
        clean_cache_shape = clean_fwd_cache[hook_point].shape
        if len(clean_cache_shape) == 2:
            corrupt_cache[hook_point] = mean_cache[hook_point][:, :clean_cache_shape[0]]
        else:
            corrupt_cache[hook_point] = mean_cache[hook_point][:, :clean_cache_shape[1]]
    corrupt_cache = ActivationCache(corrupt_cache, model)
    return corrupt_cache


# def compute_attribution_patching_scores_for_hook_points(
#     clean_fwd_cache: ActivationCache,
#     corrupt_fwd_cache: ActivationCache | None,
#     bwd_cache: ActivationCache,
#     hook_points: list[str],
#     corrupt_bwd_cache: ActivationCache | None = None,
#     zero_ablation: bool = False,
#     use_corrupt_bwd_cache: bool = False,
# ):
#     activations = torch.stack(
#         [clean_fwd_cache[hook_point].squeeze(0) for hook_point in hook_points]
#     )
#     if not zero_ablation:
#         corrupt_activations = torch.stack(
#             [corrupt_fwd_cache[hook_point].squeeze(0) for hook_point in hook_points]
#         )

#     gradients = torch.stack(
#         [bwd_cache[hook_point].squeeze(0) for hook_point in hook_points[:-1]]
#     )

#     attributions = []

#     for layer in range(len(hook_points) - 1):
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         if zero_ablation:
#             attributions.append(gradients[layer] * activations[layer])
#         else:
#             corrupt_acts = corrupt_activations[layer].to(activations[layer].device)[
#                 : activations[layer].shape[0]
#             ]
#             attributions.append(
#                 gradients[layer].to(activations[layer].device)
#                 * (corrupt_acts - activations[layer])
#             )
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#     return torch.stack(attributions)
