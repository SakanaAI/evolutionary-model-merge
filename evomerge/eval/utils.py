import logging
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from .metrics import rouge_ja, LanguageDetector


logger = logging.getLogger(__name__)


@dataclass
class EvalOutput:
    metrics: Dict[str, float]
    results: List[Dict[str, Any]]


def dict_collation_fn(samples: List) -> Dict:
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        result[key] = list(batched[key])

    del samples
    del batched
    return result


def flatten_list(
    results: List[Dict[str, List[Union[str, bool]]]]
) -> Dict[str, List[Union[str, bool]]]:
    flatten_results = {}
    for res in results:
        for k in res:
            if k not in flatten_results:
                flatten_results[k] = res[k]
            else:
                flatten_results[k].extend(res[k])
    return flatten_results


def _evaluate(
    model, example: Dict[str, List[Any]]
) -> Dict[str, List[Union[str, bool]]]:
    question = example["question"]
    answer = example["answer"]
    image = example["image"]

    # generate responses
    resps = model(text=question, image=image)
    return {
        "question": question,
        "answer": answer,
        "prediction": resps,
    }


def compute_score(results: Dict[str, List[Any]], lang_detect=None) -> Dict[str, float]:
    res_dict = rouge_ja(refs=results["answer"], preds=results["prediction"])
    # detect Japanese by fasttext and replace empty string if it's not Ja
    if lang_detect:
        preds = []
        for answer, pred in zip(results["answer"], results["prediction"]):
            # if answer is English, pass
            if lang_detect(answer).get("__label__ja", 0.0) >= 0.5:
                res = lang_detect(pred)
                if res.get("__label__ja", 0.0) < 0.5:
                    pred = ""
            preds.append(pred)
        res_dict_ja = rouge_ja(refs=results["answer"], preds=preds)
        res_dict_ja = {f"{k}_ja": v for k, v in res_dict_ja.items()}
        res_dict.update(res_dict_ja)
    return res_dict


def evaluate(
    name: str,
    model: torch.nn.Module,
    dataset: Dataset,
    loader_kwargs: dict,
    lang_detect: Optional[LanguageDetector] = None,
):
    results = []
    dataloader = DataLoader(dataset, collate_fn=dict_collation_fn, **loader_kwargs)

    for example in tqdm(dataloader, desc=f"Evaluating {name}"):
        res = _evaluate(model, example)
        results.append(res)
    results = flatten_list(results)
    metrics = compute_score(results, lang_detect=lang_detect)
    return EvalOutput(metrics=metrics, results=results)
