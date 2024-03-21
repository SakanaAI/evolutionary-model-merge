# https://huggingface.co/stabilityai/japanese-stable-vlm
import logging
from typing import List, Union, Optional

from PIL import Image

import torch
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers import AutoImageProcessor, AutoModelForVision2Seq, AutoTokenizer

from .prompt_templates import JSVLM_TEMPLATE
from .utils import set_template, set_model_kwargs, build_prompt
from ..utils import default

logger = logging.getLogger(__name__)


class JSVLM(nn.Module, ModuleUtilsMixin):
    default_template = JSVLM_TEMPLATE
    default_generation_config = {
        "do_sample": False,
        "num_beams": 5,
        "max_new_tokens": 128,
        "min_length": 1,
        "repetition_penalty": 1.5,
    }

    def __init__(
        self,
        model_path: str = "stabilityai/japanese-stable-vlm",
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
            AutoModelForVision2Seq.from_pretrained(
                model_path, trust_remote_code=True, **model_kwargs
            )
            .eval()
            .to((device))
        )
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def forward(
        self,
        text: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image]],
        **generation_config,
    ) -> List[str]:
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
        inputs = self.processor(images=image, return_tensors="pt")
        text_encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        inputs.update(text_encoding)
        # generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs.to(self.device, dtype=self.dtype),
                **generation_config,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_text = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        generated_text = [text.strip() for text in generated_text]
        return generated_text
