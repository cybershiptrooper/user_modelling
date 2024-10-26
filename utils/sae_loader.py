import sae_lens as sl
import yaml
from pathlib import Path
from typing import Literal
import torch


def view_sae_yaml():
    """
    View the sae_lens pretrained saes yaml file as a dictionary.
    """
    root_dir = Path(sl.__file__).parent
    sae_path = root_dir / "pretrained_saes.yaml"
    with open(sae_path, "r") as f:
        sae_dict = yaml.safe_load(f)
    return sae_dict


def load_sae_id(
    release: str,
    id: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
) -> sl.SAE:
    sae = sl.SAE.from_pretrained(
        release=release,
        sae_id=id,
        device=device,
    )[0].to(dtype)
    return sae


def load_gemma_saes(
    params: Literal["2b", "9b"],
    layers: list[int],
    width: int = 16,
    type: Literal["res", "mlp", "attn"] = "res",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[int, sl.SAE]:
    release = f"gemma-scope-{params}-pt-{type}-canonical"
    saes = {}
    for layer in layers:
        saes[layer] = load_sae_id(
            release, f"layer_{layer}/width_{width}k/canonical", device, dtype
        )
    return saes
