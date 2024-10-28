import sae_lens as sl
from typing import Literal
import torch
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tabulate import tabulate
from tqdm.auto import tqdm


def get_sae_yaml() -> dict:
    """
    View the sae_lens pretrained saes yaml file as a dictionary.
    """
    return get_pretrained_saes_directory()


def view_sae_yaml() -> None:
    metadata_rows = [
        [data.model, data.release, data.repo_id, len(data.saes_map)]
        for data in get_pretrained_saes_directory().values()
    ]
    print(
        tabulate(
            metadata_rows,
            headers=["model", "release", "repo_id", "n_saes"],
            tablefmt="simple_outline",
        )
    )


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
    instruction_tuned: bool = False,
    type: Literal["res", "mlp", "attn"] = "res",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[int, sl.SAE]:
    params = params + "-it" if instruction_tuned else params
    release = f"gemma-scope-{params}-pt-{type}-canonical"
    saes = {}
    for layer in tqdm(layers, desc="Loading SAEs"):
        saes[layer] = load_sae_id(
            release, f"layer_{layer}/width_{width}k/canonical", device, dtype
        )
    return saes
