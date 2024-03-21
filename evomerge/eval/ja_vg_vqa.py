"""
Japanese VQA of Visual Genome 
https://github.com/yahoojapan/ja-vg-vqa
"""

from typing import Optional

from datasets import load_dataset

from .metrics import LanguageDetector
from .utils import EvalOutput, evaluate
from ..utils import default


def extract_qa(example):
    qa_list = example["qas"]
    # TODO: for now, always take the first example
    # should we evaluate everything? Or, take one example randomely?
    qa = qa_list[0]
    example["question"] = qa["question"]
    example["answer"] = qa["answer"]
    return example


class JaVGVQA:
    name = "JA-VG-VQA-500"
    dataset_path = "SakanaAI/JA-VG-VQA-500"
    dataset_split = "test"

    def __init__(
        self,
        verbose: bool = False,
        loader_kwargs: Optional[dict] = None,
        strict_japanese: bool = True,
    ):
        self.verbose = verbose
        self.loader_kwargs = default(loader_kwargs, {})

        dataset = load_dataset(path=self.dataset_path, split=self.dataset_split)
        # extract qa
        dataset = dataset.map(extract_qa)
        # filter column
        self.dataset = dataset.select_columns(["question", "answer", "image"])
        self.lang_detect = None
        if strict_japanese:
            self.lang_detect = LanguageDetector()

    def __call__(self, model) -> EvalOutput:
        return evaluate(
            name=self.name,
            model=model,
            dataset=self.dataset,
            loader_kwargs=self.loader_kwargs,
            lang_detect=self.lang_detect,
        )
