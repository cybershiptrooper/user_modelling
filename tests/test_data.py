from utils.data import BestOrWorstConstantAnswer


def test_best_or_worst():
    template = BestOrWorstConstantAnswer(
        user_prompt="What is the capital of France?",
        options=["Paris", "London"],
    )
    best_prompt = template.get_clean_prompt()

    check = """<start_of_turn>system
Select the **best** possible option from the choices the user provides. Output only the letter of the option you choose.
<start_of_turn>user
What is the capital of France?
a. Paris
b. London
<start_of_turn>model"""

    assert best_prompt == check

    worst_prompt = template.get_counterfactual_prompt()

    check = """<start_of_turn>System
Select the **worst** possible option from the choices the user provides. Output only the letter of the option you choose.
<start_of_turn>user
What is the capital of France?
a. London
b. Paris
<start_of_turn>model"""

    assert worst_prompt == check
