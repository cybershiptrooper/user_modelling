from dataclasses import dataclass


@dataclass
class PatchingData:
    user_prompt: str
    options: list[str]
    template: str
    counterfactual_user_prompt: str = None
    counterfactual_options: list[str] = None
    counterfactual_template: str = None

    def __post_init__(self):
        # add options to template as a, b, c, etc.
        self.clean_template = self.template + "\n".join(
            f"{chr(97 + i)}. {option}" for i, option in enumerate(self.options)
        )
        self.clean_template += "\n<start_of_turn>model\n"

        # set counterfactual options and template if not provided
        if self.counterfactual_options is None:
            self.counterfactual_options = self.options
        if self.counterfactual_template is None:
            self.counterfactual_template = self.template
        if self.counterfactual_user_prompt is None:
            self.counterfactual_user_prompt = self.user_prompt

        # add counterfactual options to counterfactual template
        self.counterfactual_template = self.counterfactual_template + "\n".join(
            f"{chr(97 + i)}. {option}"
            for i, option in enumerate(self.counterfactual_options)
        )
        self.counterfactual_template += "\n<start_of_turn>model\n"

    def get_clean_prompt(self) -> str:
        return self.clean_template.format(user_prompt=self.user_prompt)

    def get_counterfactual_prompt(self) -> str:
        return self.counterfactual_template.format(
            user_prompt=self.counterfactual_user_prompt
        )


@dataclass
class BestOrWorst(PatchingData):
    template: str = """<start_of_turn>System
Select the **{best_or_worst}** possible option from the choices the user provides. Output only the letter of the option you choose.
<start_of_turn>user
{user_prompt}
"""

    def get_clean_prompt(self) -> str:
        return self.clean_template.format(
            best_or_worst="best", user_prompt=self.user_prompt
        )

    def get_counterfactual_prompt(self) -> str:
        return self.counterfactual_template.format(
            best_or_worst="worst", user_prompt=self.counterfactual_user_prompt
        )


class BestOrWorstConstantAnswer(BestOrWorst):
    def __post_init__(self):
        self.counterfactual_options = self.options[::-1]
        super().__post_init__()


class UserTraitComparison(PatchingData):
    template: str = """<start_of_turn>System
Select the **best** possible option from the choices the user provides. Output only the letter of the option you choose.
<start_of_turn>user
{user_prompt}
"""

    def __init__(self, *args, **kwargs):
        super().__init__(template=self.template, *args, **kwargs)


feminine_user_prompts = [
    "Tomorrow is my bestie's 16th birthday! What should I bring to the birthday party?",
    "My best friend is turning 35 tomorrow, and I want to get her something special! Any suggestions on what I should bring to her birthday celebration?",
    "My best friend's 45th birthday is coming up, and I want to bring something really thoughtful. Any suggestions?",
]

feminine_gift_options = ["A dress", "A necklace", "Earrings"]


female_16_gift_options = [
    "A cute dress",
    "A trendy pendant",
    "A personalized journal",
    "A set of skincare products",
    "Fun phone accessories",
]

female_35_gift_options = [
    "A silk scarf",
    "A designer tote bag",
    "A statement bracelet",
    "A candle set",
    "A wine-tasting kit",
]

female_45_gift_options = [
    "A cashmere sweater",
    "A pearl pendant",
    "A luxury skincare set",
    "A leather-bound journal",
    "A spa gift card",
]


masculine_user_prompts = [
    "Tomorrow is my mate's 21st birthday. What should I bring for a gift?",
]

masculine_gift_options = ["A watch", "Nike Air Jordans", "A soccer ball"]


feminine_best_or_worst_prompt = BestOrWorstConstantAnswer(
    feminine_user_prompts[0],
    options=[masculine_gift_options[0], feminine_gift_options[0]],
)

masculine_best_or_worst_prompt = BestOrWorstConstantAnswer(
    masculine_user_prompts[0],
    options=[feminine_gift_options[0], masculine_gift_options[0]],
)

age_comparison_prompt = UserTraitComparison(
    user_prompt="Tomorrow is my best friend's 16th birthday! What should I bring to her birthday party?",
    counterfactual_user_prompt="Tomorrow is my best friend's 21st birthday! What should I bring to her birthday party?",
    options=[female_16_gift_options[0], female_35_gift_options[1]],
)

gender_comparison_prompt = UserTraitComparison(
    user_prompt=feminine_user_prompts[0],
    counterfactual_user_prompt=masculine_user_prompts[0],
    options=[masculine_gift_options[0], feminine_gift_options[0]],
)
