import os
import re

from rouge_score import rouge_scorer, scoring
from fasttext.FastText import _FastText


def mean(l):
    return sum(l) / len(l)


def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types)

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


class MecabTokenizer:
    def __init__(self) -> None:
        from fugashi import Tagger

        self.tagger = Tagger("-Owakati")

    def normalize_answer(self, text):
        """Lower case text, remove punctuation and extra whitespace, etc."""
        import emoji
        import neologdn

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_emoji(text):
            text = "".join(["" if emoji.is_emoji(c) else c for c in text])
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                "\U00002702-\U000027B0"
                "]+",
                flags=re.UNICODE,
            )
            return emoji_pattern.sub(r"", text)

        text = remove_emoji(text)
        # see neologdn docs for details, but handles things like full/half width variation
        text = neologdn.normalize(text)
        text = white_space_fix(text)
        return text

    def tokenize(self, text):
        return self.tagger.parse(self.normalize_answer(text)).split()


def rouge_ja(refs, preds):
    """This uses a MeCab tokenizer for Japanese text."""
    tokenizer = MecabTokenizer()
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    # mecab-based rouge
    scorer = rouge_scorer.RougeScorer(
        rouge_types,
        tokenizer=tokenizer,
    )

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


class LanguageDetector:
    def __init__(self):
        model_path = os.environ.get("LID176FTZ_PATH", "lid.176.ftz")
        self.model = _FastText(model_path)

    def __call__(self, text: str) -> dict:
        return dict(zip(*self.model.predict(text.replace("\n", ""), k=-1)))
