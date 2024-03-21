import os
import argparse
import gc
import json
import logging
import os
from dataclasses import asdict

import torch

from evomerge import instantiate_from_config, load_config, set_seed

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="config path")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    # validation
    if args.output_path is None:
        args.output_path = (
            os.path.splitext(os.path.basename(args.config_path))[0] + ".json"
        )
        args.output_path = f"results/{args.output_path}"
        os.makedirs("results", exist_ok=True)
    assert args.output_path.endswith(".json"), "`output_path` must be json file"
    return args


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    config = load_config(args.config_path)
    logger.info(f"Config:\n{json.dumps(config, indent=2, ensure_ascii=False)}")
    set_seed(42)

    # 1. load model (it's already moved to device)
    model = instantiate_from_config(config["model"])
    logger.info(f"Model: {model.__class__.__name__}")

    eval_configs = config["eval"]
    if isinstance(eval_configs, dict):
        eval_configs = [eval_configs]

    results = {}
    for eval_config in eval_configs:
        # 2. load evaluator
        evaluator = instantiate_from_config(eval_config)
        logger.info(f"Evaluator: {evaluator.__class__.__name__}")
        # 3. Run!
        outputs = evaluator(model)
        logger.info(f"Result:\n{outputs.metrics}")
        results[evaluator.name] = asdict(outputs)
        del evaluator
        torch.cuda.empty_cache()
        gc.collect()

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
