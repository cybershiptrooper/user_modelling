import torch
from transformer_lens import ActivationCache
from sae_lens import SAE
import pandas as pd
from jaxtyping import Float
import transformer_lens as tl
from utils import get_cache_fwd_and_bwd
from typing import Callable, Literal
import einops


def compute_sae_activations_and_attributions(
    saes: list[SAE],
    fwd_cache: ActivationCache | dict,
    bwd_cache: ActivationCache | dict,
    hook_points: list[str],
):
    all_layers = list(saes.keys())
    assert len(hook_points) == len(saes), "hook_points must be the same length as saes"
    residual_streams = torch.stack(
        [
            fwd_cache[hook_points[layer_index]].squeeze(0)
            for layer_index, layer in enumerate(all_layers)
        ]
    )
    gradient_residuals = torch.stack(
        [
            bwd_cache[hook_points[layer_index]].squeeze(0)
            for layer_index, layer in enumerate(all_layers)
        ]
    )

    sae_activations_list = []
    sae_attributions_list = []
    for layer_index, layer in enumerate(all_layers):
        _, sae_cache = saes[layer].run_with_cache(
            residual_streams[layer_index].float().to("cuda")
        )
        sae_activations = sae_cache["hook_sae_acts_post"]
        sae_attributions = (
            gradient_residuals[layer_index].float().to("cuda") @ saes[layer].W_dec.T
        ) * sae_activations
        sae_activations_list.append(sae_activations)
        sae_attributions_list.append(sae_attributions)

    sae_activations = torch.stack(sae_activations_list).to("cpu")
    sae_attributions = torch.stack(sae_attributions_list).to("cpu")

    return sae_activations, sae_attributions


def get_top_latent_contributions(
    per_latent_contribution: Float[torch.Tensor, "batch layer pos d_sae"],  # noqa: F722
    top_k: int = 5,
    absolute: bool = True,
):
    results = []
    for layer in range(per_latent_contribution.shape[0]):
        latent_contributions = per_latent_contribution[layer, :]
        abs_latent_contributions = latent_contributions.abs()
        if absolute:
            top_k_latents = abs_latent_contributions.topk(top_k).indices.tolist()
        else:
            top_k_latents = latent_contributions.topk(top_k).indices.tolist()
        for latent_idx in top_k_latents:
            results.append(
                {
                    "latent_idx": latent_idx,
                    "layer": layer,
                    "contribution": per_latent_contribution[layer, latent_idx].item(),
                    "abs_contribution": abs_latent_contributions[latent_idx].item(),
                }
            )
    return pd.DataFrame(results)


def get_top_changes_in_latent_contributions(
    per_latent_contribution_before: Float[torch.Tensor, "batch layer pos d_sae"],  # noqa: F722
    per_latent_contribution_after: Float[torch.Tensor, "batch layer pos d_sae"],  # noqa: F722
    top_k: int = 5,
    absolute: bool = True,
    normalize: bool = False,
):
    results = []
    for layer in range(per_latent_contribution_before.shape[0]):
        latent_contributions_before = per_latent_contribution_before[layer, :]
        latent_contributions_after = per_latent_contribution_after[layer, :]

        change_in_contributions = (
            latent_contributions_after - latent_contributions_before
        )
        if normalize:
            change_in_contributions /= (
                latent_contributions_before.abs()
                + latent_contributions_after.abs()
                + 1e-8
            ) / 2
        abs_change_in_contributions = change_in_contributions.abs()
        if absolute:
            top_k_changes = abs_change_in_contributions.topk(top_k).indices.tolist()
        else:
            top_k_changes = change_in_contributions.topk(top_k).indices.tolist()
        for latent_idx in top_k_changes:
            results.append(
                {
                    "latent_idx": latent_idx,
                    "layer": layer,
                    "change": change_in_contributions[latent_idx].item(),
                    "abs_change": abs_change_in_contributions[latent_idx].item(),
                    "contribution_before": latent_contributions_before[
                        latent_idx
                    ].item(),
                    "contribution_after": latent_contributions_after[latent_idx].item(),
                }
            )
    return pd.DataFrame(results)


def compute_attribution_patching_scores_for_hook_points(
    clean_fwd_cache: ActivationCache,
    corrupt_fwd_cache: ActivationCache | None,
    bwd_cache: ActivationCache,
    hook_points: list[str],
    zero_ablation: bool = False,
):
    activations = torch.stack(
        [clean_fwd_cache[hook_point].squeeze(0) for hook_point in hook_points]
    )
    if not zero_ablation:
        corrupt_activations = torch.stack(
            [corrupt_fwd_cache[hook_point].squeeze(0) for hook_point in hook_points]
        )

    gradients = torch.stack(
        [bwd_cache[hook_point].squeeze(0) for hook_point in hook_points]
    )

    attributions = []

    for layer in range(len(hook_points)):
        if zero_ablation:
            attributions.append(gradients[layer] * activations[layer])
        else:
            corrupt_acts = corrupt_activations[layer].to(activations[layer].device)
            attributions.append(gradients[layer] * (activations[layer] - corrupt_acts))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.stack(attributions)


def compute_model_attribution_patching_scores(
    model: tl.HookedTransformer,
    clean_prompt: str,
    corrupt_prompt: str,
    metric: Callable,
    hook_points: list[str],
    zero_ablation: bool = False,
):
    _, clean_fwd_cache, clean_bwd_cache = get_cache_fwd_and_bwd(
        model, clean_prompt, metric=metric, hook_points=hook_points
    )
    if not zero_ablation:
        _, corrupt_fwd_cache, corrupt_bwd_cache = get_cache_fwd_and_bwd(
            model, corrupt_prompt, metric=metric, hook_points=hook_points
        )
    else:
        corrupt_fwd_cache = None

    attrs = compute_attribution_patching_scores_for_hook_points(
        clean_fwd_cache,
        corrupt_fwd_cache,
        clean_bwd_cache,
        hook_points,
        zero_ablation,
    )

    del corrupt_fwd_cache, corrupt_bwd_cache, clean_fwd_cache, clean_bwd_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return attrs


def compute_attention_attribution_for_prompt(
    model: tl.HookedTransformer,
    prompt: str,
    metric: Callable,
    return_acts: bool = False,
):
    attn_hook_points = [
        f"blocks.{layer}.attn.hook_pattern" for layer in range(model.cfg.n_layers)
    ]
    _, fwd_cache, bwd_cache = get_cache_fwd_and_bwd(
        model, prompt, metric=metric, hook_points=attn_hook_points
    )
    acts = fwd_cache.stack_activation("attn").squeeze()
    grads = bwd_cache.stack_activation("attn").squeeze()
    if return_acts:
        return acts, (acts * grads).squeeze()
    return (acts * grads).squeeze()


def stack_head_vector_from_cache(
    model, cache, activation_name: Literal["q", "k", "v", "z"]
):
    """Stacks the head vectors from the cache from a specific activation (key, query, value or mixed_value (z)) into a single tensor."""
    stacked_head_vectors = torch.stack(
        [cache[activation_name, l] for l in range(model.cfg.n_layers)], dim=0
    )
    stacked_head_vectors = einops.rearrange(
        stacked_head_vectors,
        "layer batch pos head_index d_head -> (layer head_index) batch pos d_head",
    )
    return stacked_head_vectors


def attr_patch_head_vector(
    model: tl.HookedTransformer,
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
    activation_name: Literal["q", "k", "v", "z"],
):
    HEAD_NAMES = [
        f"L{l}H{h}"  # noqa
        for l in range(model.cfg.n_layers)  # noqa
        for h in range(model.cfg.n_heads)  # noqa
    ]
    labels = HEAD_NAMES

    clean_head_vector = stack_head_vector_from_cache(model, clean_cache, activation_name)
    corrupted_head_vector = stack_head_vector_from_cache(
        model, corrupted_cache, activation_name
    )
    corrupted_grad_head_vector = stack_head_vector_from_cache(
        model, corrupted_grad_cache, activation_name
    )
    head_vector_attr = einops.reduce(
        corrupted_grad_head_vector * (clean_head_vector - corrupted_head_vector),
        "component batch pos d_head -> component pos",
        "sum",
    )
    print(
        len(labels),
        head_vector_attr.shape,
        clean_head_vector.shape,
        corrupted_head_vector.shape,
        corrupted_grad_head_vector.shape,
    )
    return head_vector_attr, labels
