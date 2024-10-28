from huggingface_hub import hf_hub_download
import torch


def load_probe(
    name: str,
    cache_dir: str = "./cache",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    weights_file = hf_hub_download(
        "thorsley/user_model_probes",
        subfolder="",
        filename=f"collected_{name}_probe_weights.pt",
        cache_dir=cache_dir,
        force_download=False,
    )

    bias_file = hf_hub_download(
        "thorsley/user_model_probes",
        subfolder="",
        filename=f"collected_{name}_probe_biases.pt",
        cache_dir=cache_dir,
        force_download=False,
    )

    weights = torch.load(weights_file, map_location=device)
    biases = torch.load(bias_file, map_location=device)

    return weights, biases