import torch
from jaxtyping import Float


def logit_diff_metric(
    logits: Float[torch.Tensor, "batch pos d_model"],  # noqa
    correct_answer_logit_idx: int,
    wrong_answer_logit_idx: int,
) -> float:
    return (
        logits[:, -1, correct_answer_logit_idx] - logits[:, -1, wrong_answer_logit_idx]
    ).mean()


def logits_to_ave_logit_diff(
    logits: Float[torch.Tensor, "batch seq d_vocab"],
    answer_tokens: Float[torch.Tensor, "batch 2"],
    per_prompt: bool = False,
) -> Float[torch.Tensor, "*batch"]:
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    # SOLUTION
    # Only the final logits are relevant for the answer
    final_logits: Float[torch.Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[torch.Tensor, "batch 2"] = final_logits.gather(
        dim=-1, index=answer_tokens
    )
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


def ioi_metric(
    logits: Float[torch.Tensor, "batch seq d_vocab"],
    answer_tokens: Float[torch.Tensor, "batch 2"],
    corrupted_logit_diff: float,
    clean_logit_diff: float,
) -> float:
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.
    """
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (patched_logit_diff - corrupted_logit_diff) / (
        clean_logit_diff - corrupted_logit_diff
    )
