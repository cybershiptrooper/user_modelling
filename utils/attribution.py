import torch
from transformer_lens import HookedTransformer, ActivationCache
from sae_lens import SAE
import pandas as pd
from jaxtyping import Float


def compute_sae_activations_and_attributions(
    model: HookedTransformer,
    saes: list[SAE],
    fwd_cache: ActivationCache | dict,
    bwd_cache: ActivationCache | dict,
    hook_points: list[str],
):
    all_layers = list(range(model.cfg.n_layers))
    residual_streams = torch.stack(
        [fwd_cache[hook_points[layer]].squeeze(0) for layer in all_layers]
    )
    gradient_residuals = torch.stack(
        [bwd_cache[hook_points[layer]].squeeze(0) for layer in all_layers]
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
):
    results = []
    for layer in range(per_latent_contribution_before.shape[0]):
        latent_contributions_before = per_latent_contribution_before[layer, :]
        latent_contributions_after = per_latent_contribution_after[layer, :]

        change_in_contributions = (
            latent_contributions_after - latent_contributions_before
        )
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
                    "contribution_before": latent_contributions_before[latent_idx].item(),
                    "contribution_after": latent_contributions_after[latent_idx].item(),
                }
            )
    return pd.DataFrame(results)
