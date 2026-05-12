# Adapted from https://github.com/openai/simple-evals/

"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re
from typing import Optional

import pandas

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    format_multichoice_question,
)

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


def _stratified_sample(examples, num_examples, key="Subject", seed=0):
    """Sample proportionally by `key` so subject coverage is preserved."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for ex in examples:
        buckets[ex.get(key, "")].append(ex)
    rng = random.Random(seed)
    sampled = []
    n_subjects = len(buckets)
    per_subject = max(1, num_examples // n_subjects)
    for bucket in buckets.values():
        sampled.extend(rng.sample(bucket, min(per_subject, len(bucket))))
    # top up to exactly num_examples if rounding left gaps
    remainder = [ex for ex in examples if ex not in sampled]
    rng.shuffle(remainder)
    sampled.extend(remainder[: max(0, num_examples - len(sampled))])
    return sampled[:num_examples]


class MMLUEval(Eval):
    def __init__(self, filename: str, num_examples: Optional[int], num_threads: int):
        if "://" in filename:
            df = pandas.read_csv(filename, storage_options={"timeout": 30})
        else:
            df = pandas.read_csv(filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = _stratified_sample(examples, num_examples)
        self.examples = examples
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(row), role="user"
                )
            ]
            response_text = sampler(prompt_messages)
            response_text = response_text or ""
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = subject2category.get(row["Subject"], "other")
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
