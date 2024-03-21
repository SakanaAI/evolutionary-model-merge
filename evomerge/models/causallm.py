import logging
from typing import List, Union, Optional
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin
from vllm import LLM, SamplingParams

from .prompt_templates import JA_ALPACA_COT_TEMPLATE
from .utils import set_template, build_prompt
from ..utils import default

logger = logging.getLogger(__name__)


class CausalLMWithvLLM(nn.Module, ModuleUtilsMixin):
    default_generation_config = {
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }
    default_template = JA_ALPACA_COT_TEMPLATE

    def __init__(
        self,
        model_path: str = None,
        template: Optional[str] = None,
        verbose: bool = False,
        model_kwargs: Optional[dict] = None,
        generation_config: Optional[dict] = None,
    ):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.verbose = verbose
        self.template = set_template(self.default_template, template)
        self.model_kwargs = default(model_kwargs, {})
        self.generation_config = default(
            generation_config, self.default_generation_config
        )
        self.model = LLM(model=model_path, **self.model_kwargs)
        self.post_init()

    def post_init(self):
        stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
        self.generation_config = SamplingParams(
            **self.generation_config, stop=stop_tokens
        )

    def forward(self, text: Union[str, List[str]]) -> List[str]:
        text = build_prompt(text, self.template)
        if self.verbose:
            logger.info(
                "Sample of actual inputs:\n" + "-" * 100 + f"\n{text[0]}\n" + "-" * 100
            )
        outputs = self.model.generate(
            prompts=text, sampling_params=self.generation_config
        )
        generated_text = [output.outputs[0].text for output in outputs]
        return generated_text
