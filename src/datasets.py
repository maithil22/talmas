"""
Dataset adapters for multi-dataset evaluation.

To add a new dataset:
  1. Write any adapter-specific helper functions (extract_fn, match_fn, etc.)
  2. Create a DatasetAdapter instance
  3. Call register_dataset(adapter)
"""

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> Optional[str]:
    m = re.search(r"####\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    m = re.search(r"[Tt]he answer is\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract content of the last \\boxed{...}, handling nested braces."""
    results = []
    i = 0
    marker = r"\boxed{"
    while True:
        start = text.find(marker, i)
        if start == -1:
            break
        content_start = start + len(marker)
        depth = 0
        j = content_start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    results.append(text[content_start:j].strip())
                    break
                depth -= 1
            j += 1
        i = start + 1
    return results[-1] if results else None


def numeric_match(pred: Optional[str], gold: Optional[str]) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return float(pred) == float(str(gold).replace(",", ""))
    except ValueError:
        return pred.strip() == str(gold).strip()


def math_string_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Normalize whitespace, compare strings, then try numeric fallback."""
    if pred is None or gold is None:
        return False
    pred_n = re.sub(r"\s+", "", pred)
    gold_n = re.sub(r"\s+", "", str(gold))
    if pred_n == gold_n:
        return True
    try:
        return float(pred_n) == float(gold_n)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# DatasetAdapter
# ---------------------------------------------------------------------------

@dataclass
class DatasetAdapter:
    """All dataset-specific behavior for one benchmark."""

    name: str
    hf_path: str
    hf_config: Optional[str]
    question_field: str
    answer_field: str
    few_shot_text: str
    default_split: str = "test"
    # Callable overrides — None means use the built-in default
    extract_fn: Optional[Callable[[str], Optional[str]]] = None
    gold_extract_fn: Optional[Callable[[str], Optional[str]]] = None
    match_fn: Optional[Callable] = None
    prompt_fn: Optional[Callable] = None

    def load(
        self,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        indices: Optional[List[int]] = None,
    ):
        """Load and slice the dataset. indices takes priority over max_samples."""
        from datasets import load_dataset as hf_load

        ds = hf_load(self.hf_path, self.hf_config, split=split or self.default_split)
        if indices is not None:
            ds = ds.select(indices)
        elif max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds

    def get_question(self, example: dict) -> str:
        return example[self.question_field]

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract predicted answer from model output text."""
        fn = self.extract_fn if self.extract_fn is not None else extract_gsm8k_answer
        return fn(text)

    def extract_gold(self, example: dict) -> Optional[str]:
        """Extract gold answer from a dataset example dict."""
        raw = example[self.answer_field]
        if self.gold_extract_fn is not None:
            return self.gold_extract_fn(raw)
        # Default: apply the same extractor used for model output
        return self.extract_answer(raw)

    def answers_match(self, pred: Optional[str], gold: Optional[str]) -> bool:
        fn = self.match_fn if self.match_fn is not None else numeric_match
        return fn(pred, gold)

    def build_prompt(self, question: str, is_instruct: bool) -> str:
        if self.prompt_fn is not None:
            return self.prompt_fn(question, is_instruct, self.few_shot_text)
        return _default_prompt(question, is_instruct, self.few_shot_text)


# ---------------------------------------------------------------------------
# Default prompt builders
# ---------------------------------------------------------------------------

def _default_prompt(question: str, is_instruct: bool, few_shot_text: str) -> str:
    if is_instruct:
        system = (
            "Solve the following math problem step by step. "
            "At the end, state 'The answer is X.' where X is the numeric answer."
        )
        return (
            f"<|system|>\n{system}\n"
            f"<|user|>\n{few_shot_text.strip()}\n\nQuestion: {question}\nAnswer:"
            f"<|assistant|>\n"
        )
    return few_shot_text + f"Question: {question}\nAnswer:"


def _math_prompt(question: str, is_instruct: bool, few_shot_text: str) -> str:
    if is_instruct:
        system = (
            "Solve the following math problem step by step. "
            "Write your final answer inside \\boxed{}."
        )
        return (
            f"<|system|>\n{system}\n"
            f"<|user|>\n{few_shot_text.strip()}\n\nProblem: {question}\nSolution:"
            f"<|assistant|>\n"
        )
    return few_shot_text + f"Problem: {question}\nSolution:"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, DatasetAdapter] = {}


def register_dataset(adapter: DatasetAdapter) -> None:
    DATASET_REGISTRY[adapter.name] = adapter


def get_adapter(name: str) -> DatasetAdapter:
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name]


def list_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY)


# ---------------------------------------------------------------------------
# GSM8K adapter
# ---------------------------------------------------------------------------

_GSM8K_FEW_SHOT = """\
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

"""

register_dataset(DatasetAdapter(
    name="gsm8k",
    hf_path="gsm8k",
    hf_config="main",
    question_field="question",
    answer_field="answer",
    few_shot_text=_GSM8K_FEW_SHOT,
    default_split="test",
    extract_fn=extract_gsm8k_answer,
    gold_extract_fn=extract_gsm8k_answer,
    match_fn=numeric_match,
))


# ---------------------------------------------------------------------------
# MATH adapter  (qwedsacf/competition_math — fields: problem, solution, level, type)
# Only a 'train' split (12,500 rows). Answer embedded in 'solution' as \boxed{}.
# ---------------------------------------------------------------------------

_MATH_FEW_SHOT = """\
Problem: Simplify $\\sqrt{245}$.
Solution: We factor $245 = 5 \\times 7^2$, so $\\sqrt{245} = 7\\sqrt{5}$. The answer is $\\boxed{7\\sqrt{5}}$.

Problem: What is the greatest common divisor of 48 and 180?
Solution: $48 = 2^4 \\times 3$ and $180 = 2^2 \\times 3^2 \\times 5$, so $\\gcd(48,\\,180) = 2^2 \\times 3 = 12$. The answer is $\\boxed{12}$.

Problem: A right triangle has legs of length 5 and 12. What is the length of the hypotenuse?
Solution: By the Pythagorean theorem, $c = \\sqrt{5^2 + 12^2} = \\sqrt{169} = 13$. The answer is $\\boxed{13}$.

Problem: How many ways can 3 books be chosen from a shelf of 7 distinct books?
Solution: $\\binom{7}{3} = \\frac{7 \\times 6 \\times 5}{3!} = 35$. The answer is $\\boxed{35}$.

"""

register_dataset(DatasetAdapter(
    name="math",
    hf_path="qwedsacf/competition_math",
    hf_config=None,
    question_field="problem",
    answer_field="solution",  # answer embedded as \boxed{} inside the solution field
    few_shot_text=_MATH_FEW_SHOT,
    default_split="train",    # only split available
    extract_fn=extract_boxed_answer,
    gold_extract_fn=extract_boxed_answer,
    match_fn=math_string_match,
    prompt_fn=_math_prompt,
))
