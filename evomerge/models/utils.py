import logging
from typing import Optional, Union, List


import torch
from .prompt_templates import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)


STR2DTYPE = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "auto": "auto",
}


def get_output_ids(input_ids: torch.Tensor, output_ids: torch.Tensor):
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        logger.warn(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    return output_ids[:, input_token_len:]


def set_template(default_template: str, template: Optional[str] = None):
    if template is None:
        template = default_template
    if template in PROMPT_TEMPLATES:
        template = PROMPT_TEMPLATES[template]
    logger.info(f"prompt template:\n{template}")
    return template


def set_model_kwargs(model_kwargs: dict) -> dict:
    torch_dtype = model_kwargs.pop("torch_dtype", None)
    if torch_dtype is not None and isinstance(torch_dtype, str):
        model_kwargs["torch_dtype"] = STR2DTYPE[torch_dtype]
    return model_kwargs


def build_prompt(text: Union[str, List[str]], template: str) -> List[str]:
    if isinstance(text, str):
        text = [text]
    return [template.format(input=t) for t in text]
