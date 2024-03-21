import logging
from typing import List, Union, Optional

from PIL import Image

import torch
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .prompt_templates import LLAVA_MISTRAL_TEMPLATE
from .utils import set_template, set_model_kwargs, get_output_ids
from ..utils import default

logger = logging.getLogger(__name__)


def build_prompt(text: Union[str, List[str]], template: str) -> List[str]:
    if isinstance(text, str):
        text = [text]
    return [template.format(input=f"<image>\n{t}") for t in text]


class LLaVA(nn.Module, ModuleUtilsMixin):
    default_template = LLAVA_MISTRAL_TEMPLATE
    # taken from https://github.com/haotian-liu/LLaVA/blob/main/predict.py#L87
    default_generation_config = {
        "do_sample": True,
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "top_p": 1.0,
        "use_cache": True,
    }

    def __init__(
        self,
        model_path: str = "llava-hf/llava-v1.6-mistral-7b-hf",
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
            (
                LlavaForConditionalGeneration.from_pretrained(
                    model_path, **model_kwargs
                )
                .eval()
                .requires_grad_(False)
            )
            .eval()
            .to(device)
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def forward(
        self,
        text: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image]],
        **generation_config,
    ) -> List[str]:
        """
        Assume text is question string
        """
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
            text=text, images=image, padding=True, return_tensors="pt"
        )
        # generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs.to(self.device, dtype=self.dtype),
                **generation_config,
            )
        # output_ids contains input_ids as well. So, return only output_ids
        output_ids = get_output_ids(input_ids=inputs.input_ids, output_ids=output_ids)
        generated_text = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        generated_text = [text.strip() for text in generated_text]
        return generated_text
