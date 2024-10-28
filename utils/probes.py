import requests
import zipfile
from pathlib import Path
from typing import Tuple, List
import torch
import torch.nn as nn

# Label mappings
LABEL_MAPS = {
    "age": {
        "child": 0,
        "adolescent": 1,
        "adult": 2,
        "older adult": 3,
    },
    "gender": {
        "male": 0,
        "female": 1,
    },
    "socioeconomic": {"low": 0, "middle": 1, "high": 2},
    "education": {"someschool": 0, "highschool": 1, "collegemore": 2},
}

DATA_MAPS = {"age": 2, "gender": 4, "socioeconomic": 2, "education": 3}


def load_dataset(
    attribute: str,
) -> Tuple[List[str], List[int]]:
    """Load dataset for a given attribute"""

    texts = []
    labels = []
    label_map = LABEL_MAPS[attribute]

    if attribute == "education":
        data_paths = [
            Path(f"dataset/openai_{attribute}_three_classes_{i}.zip")
            for i in range(1, DATA_MAPS[attribute] + 1)
        ]
    else:
        data_paths = [
            Path(f"dataset/openai_{attribute}_{i}.zip")
            for i in range(1, DATA_MAPS[attribute] + 1)
        ]

    for data_path in data_paths:
        if not data_path.exists():
            print("Downloading dataset...")
            url = f"https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/raw/main/data/dataset/{data_path.name}"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"Download failed: HTTP {response.status_code}")

            data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(data_path, "wb") as f:
                f.write(response.content)

        # Extract if needed
        extract_path = data_path.parent / data_path.stem

        if not extract_path.exists():
            print("Extracting files...")
            with zipfile.ZipFile(data_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

        # Process txt files
        for txt_file in extract_path.glob("*.txt"):
            # Extract label from filename (e.g., "conversation_107_age_adolescent.txt" -> "adolescent")
            label = txt_file.stem.split("_")[-1]

            if label in label_map:
                with open(txt_file) as f:
                    text = f.read().strip()
                    text += (
                        f"\n \n ### Assistant: I think the {attribute} of this user is"
                    )

                texts.append(text)
                labels.append(label_map[label])

    return texts, labels


class LinearProbes(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.probe = nn.Linear(input_dim, num_classes)

        nn.init.xavier_uniform_(self.probe.weight)
        nn.init.zeros_(self.probe.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.probe(x.to(dtype=torch.float)))

    def get_grouped_params(self):
        decay = []
        no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

    @classmethod
    def from_pretrained(cls, weights: torch.Tensor, biases: torch.Tensor):
        probe = cls(weights.shape[1], weights.shape[0])
        probe.probe.weight.data = weights
        probe.probe.bias.data = biases
        return probe


def make_probes_for_each_layer(weights: torch.Tensor, biases: torch.Tensor):
    return [
        LinearProbes.from_pretrained(weights[i], biases[i])
        for i in range(weights.shape[0])
    ]