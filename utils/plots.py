from utils.neel_plotly import line, imshow
import utils.neel_utils as nutils
import torch
from transformer_lens import HookedTransformer
import plotly.graph_objects as go
import einops
from IPython.display import Markdown, display

# import einops
# import circuitsvis as cv
# from IPython.display import Markdown, display


def make_all_plots_for_residual_sae_attrs(
    model: HookedTransformer,
    sae_attrs: torch.Tensor,
    prompt: list[str],
    positions_we_care_about: list[int] | None = None,
    prepend_bos: bool = False,
):
    ALL_LAYERS = list(range(model.cfg.n_layers))
    # stats of all SAE latent attrs for each token
    x = nutils.process_tokens_index(prompt, model, prepend_bos=prepend_bos)
    line(
        sae_attrs.sum(-1),
        x=x,
        color=[str(layer) for layer in ALL_LAYERS],
        title="Per Token Attribution Scores (sum)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )
    line(
        sae_attrs.max(-1).values,
        x=x,
        color=[str(layer) for layer in ALL_LAYERS],
        title="Per Token Attribution Scores (max)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )

    if positions_we_care_about is not None:
        sae_attrs_we_care_about = sae_attrs[:, positions_we_care_about]
    else:
        sae_attrs_we_care_about = sae_attrs

    line(
        sae_attrs_we_care_about.sum(1),
        color=["Layer " + str(layer) for layer in ALL_LAYERS],
        title="Per Latent Attribution Scores (sum)",
        # labels={"x": "Tokens", "y": "Attribution Score"},
    )


def plot_probe_accuracy_histogram(accs, layer, attribute, n_bins=100):
    """plot the accuracy histogram for all prompts at a given layer"""
    # Create single figure
    fig = go.Figure()
    total_num_of_prompts = len(accs)
    # Collect normalized positions where probe is correct for this layer
    sequence_bin_matrix = torch.zeros((total_num_of_prompts, n_bins))

    for i, acc in enumerate(accs):
        correct_positions = torch.where(acc[0, layer, 1:] == 1)[0].float()
        seq_length = acc.shape[-1]
        normalized_positions = correct_positions / seq_length
        bin_width = 1.0 / n_bins
        bin_indices = (normalized_positions / bin_width).floor().long()
        # Mark 1 for each bin that has at least one correct prediction
        sequence_bin_matrix[i, bin_indices] = 1

    # Sum across sequences to get counts
    bin_counts = sequence_bin_matrix.sum(dim=0) / total_num_of_prompts
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create histogram with the correct counts
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=bin_counts,
            name=f"Layer {layer}",
            opacity=0.5,
            width=bin_width,
            textposition="auto",
            hovertemplate="Interval: %{x:.2f}, Count: %{y:.2f}",
        )
    )

    # Update layout
    fig.update_layout(
        barmode="overlay",
        title=f"Distribution of {attribute} Probe Predictions",
        xaxis_title="Normalized Position in Sequence",
        yaxis_title="Count",
        height=500,
        width=800,
        showlegend=True,
    )

    fig.show()


def plot_logit_density_histogram(probe_logits_loaded, layer, label_map, attribute):
    """plot the logit density histogram for all prompts at a given layer"""
    probe_logits_at_layer = []
    for i in range(len(probe_logits_loaded)):
        logits_for_prompt = probe_logits_loaded[i][label_map, layer, :]
        # apply inverse sigmoid
        # ln(y / (1 - y))
        logits_for_prompt = torch.log(
            logits_for_prompt / (1 - logits_for_prompt + 1e-10)
        )
        probe_logits_at_layer.append(logits_for_prompt)

    probe_logits_at_layer = torch.cat(probe_logits_at_layer)

    # Create histogram using plotly
    fig = go.Figure()
    # Split logits into negative and positive values
    logits_np = probe_logits_at_layer.cpu().numpy()
    negative_logits = logits_np[logits_np <= 0]
    positive_logits = logits_np[logits_np > 0]

    # Add positive logits histogram in default color
    fig.add_trace(
        go.Histogram(
            x=positive_logits,
            nbinsx=50,
            name=f"Layer {layer} (positive)",
            opacity=0.7,
            histnorm="probability density",
        )
    )
    # Add negative logits histogram in red
    fig.add_trace(
        go.Histogram(
            x=negative_logits,
            nbinsx=50,
            name=f"Layer {layer} (negative)",
            opacity=0.7,
            histnorm="probability density",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Logit Distribution at Layer {layer} for {attribute}",
        xaxis_title="Logit Value",
        yaxis_title="Density",
        height=500,
        width=800,
        showlegend=True,
    )

    fig.show()


def plot_head_vector_attribution(head_vector_attrs, head_vector_labels, activation_name_full, layers, heads, LABELED_TOKENS, sort = True):
    display(Markdown(f"#### {activation_name_full} Head Vector Attribution Patching"))
    imshow(
        head_vector_attrs,
        y=head_vector_labels,
        x = LABELED_TOKENS,
        yaxis="Component",
        xaxis="Position",
        title=f"{activation_name_full} Attribution Patching",
    )
    sum_head_vector_attr = einops.reduce(
        head_vector_attrs,
        "(layer head) pos -> layer head",
        "sum",
        layer=len(layers),
        head=len(heads),
    )
    sorted_sum_head_vector_attr = sum_head_vector_attr.sort(dim=-1).values
    head_indices = sum_head_vector_attr.sort(dim=-1).indices

    print(sorted_sum_head_vector_attr.shape, sum_head_vector_attr.shape)
    imshow(
        sorted_sum_head_vector_attr if sort else sum_head_vector_attr,
        yaxis="Layer",
        xaxis="Head Index",
        title=f"{activation_name_full} Attribution Patching Sum Over All Pos",
        # hover_text = [[f"Head {h}" for h in head_indices[l]] for l in range(len(layers))]
    )
    sum_pos_vector_attr = einops.reduce(
        head_vector_attrs,
        "(layer head) pos -> pos",
        "sum",
        layer=len(layers),
        head=len(heads),
    )
    line(
        sum_pos_vector_attr,
        x = LABELED_TOKENS,
        yaxis="Contribution",
        xaxis="Position",
        title=f"{activation_name_full} Attribution Patching Sum Over All Layers and Heads",
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
