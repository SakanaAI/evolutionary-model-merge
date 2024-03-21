# https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1-llava-620k
import logging
from typing import List, Union, Optional

from PIL import Image

import torch
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers import LlamaTokenizer

from ..modules.heron.video_blip import (
    VideoBlipForConditionalGeneration,
    VideoBlipProcessor,
)

from .prompt_templates import HERON_V1
from .utils import set_template, set_model_kwargs, build_prompt
from ..utils import default

logger = logging.getLogger(__name__)


class HeronV1(nn.Module, ModuleUtilsMixin):
    default_template = HERON_V1
    default_generation_config = {
        "do_sample": False,
        "temperature": 0.0,
        "max_length": 256,
        "no_repeat_ngram_size": 2,
    }

    def __init__(
        self,
        model_path: str = "turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1-llava-620k",
        template: Optional[str] = None,
        verbose: bool = False,
        device: Union[str, torch.device] = "cuda",
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ):
        super().__init__()
        self.verbose = verbose
        self.template = set_template(self.default_template, template)
        self.generation_config = default(
            generation_config, self.default_generation_config
        )
        model_kwargs = set_model_kwargs(default(model_kwargs, {}))

        self.model = (
            VideoBlipForConditionalGeneration.from_pretrained(
                model_path, ignore_mismatched_sizes=True, **model_kwargs
            )
            .eval()
            .to(device)
        )
        processor = VideoBlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1", additional_special_tokens=["▁▁"]
        )
        processor.tokenizer = tokenizer
        self.processor = processor
        self.eos_token_id_list = [
            processor.tokenizer.pad_token_id,
            processor.tokenizer.eos_token_id,
            int(tokenizer.convert_tokens_to_ids("##")),
        ]

    def forward(
        self,
        text: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image]],
        **generation_config,
    ) -> List[str]:

        eos_token_id = generation_config.pop("eos_token_id", self.eos_token_id_list)
        if len(generation_config) == 0:
            generation_config = self.generation_config
            if self.verbose:
                logger.info(
                    f"Setting generation config to default\n{generation_config}"
                )
        if not isinstance(image, list):
            image = [image]

        text = build_prompt(text, self.template)
        if self.verbose:
            logger.info(
                "Sample of actual inputs:\n" + "-" * 100 + f"\n{text[0]}\n" + "-" * 100
            )
        assert len(text) == len(image)
        inputs = self.processor(
            text=text, images=image, return_tensors="pt", padding=True
        )
        # generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs.to(self.device, dtype=self.dtype),
                **generation_config,
                eos_token_id=eos_token_id,
            )
        generated_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )
        return generated_text
