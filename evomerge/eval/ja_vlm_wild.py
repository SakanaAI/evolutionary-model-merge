from typing import Optional

from datasets import load_dataset

from .metrics import LanguageDetector
from .utils import EvalOutput, evaluate
from ..utils import default


class JaVLMBenchIntheWild:
    name = "JA-VLM-Bench-In-the-Wild"
    dataset_path = "SakanaAI/JA-VLM-Bench-In-the-Wild"
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
