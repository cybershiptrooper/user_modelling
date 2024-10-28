from utils.neel_plotly import line
import utils.neel_utils as nutils
import torch
from transformer_lens import HookedTransformer
# import einops
# import circuitsvis as cv
# from IPython.display import Markdown, display


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


# def plot_attention_attr(model, attention_attr, tokens, title=""):
#     HEAD_NAMES = [
#         f"L{l}H{h}"  # noqa
#         for l in range(model.cfg.n_layers)  # noqa
#         for h in range(model.cfg.n_heads)  # noqa
#     ]
#     HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]

#     attention_attr_pos = attention_attr.clamp(min=-1e-5)
#     attention_attr_neg = -attention_attr.clamp(max=1e-5)
#     attention_attr_signed = torch.stack([attention_attr_pos, attention_attr_neg], dim=0)
#     attention_attr_signed = einops.rearrange(
#         attention_attr_signed,
#         "sign layer head_index dest src -> (layer head_index sign) dest src",
#     )
#     attention_attr_signed = attention_attr_signed / attention_attr_signed.max()
#     attention_attr_indices = (
#         attention_attr_signed.max(-1).values.max(-1).values.argsort(descending=True)
#     )
#     # print(attention_attr_indices.shape)
#     # print(attention_attr_indices)
#     attention_attr_signed = attention_attr_signed[attention_attr_indices, :, :]
#     head_labels = [HEAD_NAMES_SIGNED[idx.item()] for idx in attention_attr_indices]

#     if title:
#         display(Markdown("### " + title))
#     display(
#         cv.attention.attention_patterns(
#             attention=attention_attr_signed,
#             tokens=tokens,
#             # attention_head_names=head_labels,
#         )
#     )
