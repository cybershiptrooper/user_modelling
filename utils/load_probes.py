from huggingface_hub import hf_hub_download
import torch


def load_probe(
    name: str,
    cache_dir: str = "./cache",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_it = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    project = "thorsley/user_model_probes" if not use_it else "thorsley/user_modelling_probes_gemma-9b-it"
    print("loading from", project)
    weights_file = hf_hub_download(
        project,
        subfolder="",
        filename=f"collected_{name}_probe_weights.pt",
        cache_dir=cache_dir,
        force_download=True,
    )

    bias_file = hf_hub_download(
        project,
        subfolder="",
        filename=f"collected_{name}_probe_biases.pt",
        cache_dir=cache_dir,
        force_download=True,
    )

    weights = torch.load(weights_file, map_location=device)
    biases = torch.load(bias_file, map_location=device)

    return weights, biases
