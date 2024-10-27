from utils.neel_plotly import line
import utils.neel_utils as nutils
import torch
from transformer_lens import HookedTransformer


def make_all_plots_for_residual_sae_attrs(
    model: HookedTransformer,
    sae_attrs: torch.Tensor,
    prompt: list[str],
    positions_we_care_about: list[int],
):
    ALL_LAYERS = list(range(model.cfg.n_layers))
    # stats of all SAE latent attrs for each token
    line(
        sae_attrs.sum(-1),
        x=nutils.process_tokens_index(prompt),
        color=[str(layer) for layer in ALL_LAYERS],
        title="Per Token Attribution Scores (sum)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )
    line(
        sae_attrs.max(-1).values,
        x=nutils.process_tokens_index(prompt),
        color=[str(layer) for layer in ALL_LAYERS],
        title="Per Token Attribution Scores (max)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )

    sae_attrs_we_care_about = sae_attrs[:, positions_we_care_about]
    line(
        sae_attrs_we_care_about.sum(1),
        color=["Layer " + str(layer) for layer in ALL_LAYERS],
        title="Per Latent Attribution Scores (sum)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )
