# 🐟 Evolutionary Optimization of Model Merging Recipes

🤗 [Models](https://huggingface.co/SakanaAI) | 👀 [Demo](https://huggingface.co/spaces/SakanaAI/EvoVLM-JP) | 📚 [Paper](https://arxiv.org/abs/2403.13187) | 📝 [Blog](https://sakana.ai/evolutionary-model-merge/) | 🐦 [Twitter](https://twitter.com/SakanaAILabs)


<div align="center">
<img src="./assets/method.gif" alt="Method" title="method">
</div>



This repository serves as a central hub for SakanaAI's [Evolutionary Model Merge](https://arxiv.org/abs/2403.13187) series, showcasing its releases and resources. It includes models and code for reproducing the evaluation presented in our paper. Look forward to more updates and additions coming soon.

## Models

### Our Models

| Model | Size | License | Source |
| :-- | --: | :-- | :-- |
| [EvoLLM-JP-v1-7B](https://huggingface.co/SakanaAI/EvoLLM-JP-v1-7B) | 7B | Microsoft Research License | [shisa-gamma-7b-v1](https://huggingface.co/augmxnt/shisa-gamma-7b-v1), [WizardMath-7B-V1.1](https://huggingface.co/WizardLM/WizardMath-7B-V1.1), [GAIR/Abel-7B-002](https://huggingface.co/GAIR/Abel-7B-002)
| [EvoLLM-JP-v1-10B](https://huggingface.co/SakanaAI/EvoLLM-JP-v1-10B) | 10B | Microsoft Research License | EvoLLM-JP-v1-7B, [shisa-gamma-7b-v1](https://huggingface.co/augmxnt/shisa-gamma-7b-v1) |
| [EvoLLM-JP-A-v1-7B](https://huggingface.co/SakanaAI/EvoLLM-JP-A-v1-7B) | 7B | Apache 2.0 | [shisa-gamma-7b-v1](https://huggingface.co/augmxnt/shisa-gamma-7b-v1), [Arithmo2-Mistral-7B](https://huggingface.co/upaya07/Arithmo2-Mistral-7B), [GAIR/Abel-7B-002](https://huggingface.co/GAIR/Abel-7B-002) |
| [EvoVLM-JP-v1-7B](https://huggingface.co/SakanaAI/EvoVLM-JP-v1-7B) | 7B | Apache 2.0 | [LLaVA-1.6-Mistral-7B](https://huggingface.co/liuhaotian/llava-v1.6-mistral-7b), [shisa-gamma-7b-v1](https://huggingface.co/augmxnt/shisa-gamma-7b-v1)




### Comparing EvoLLM-JP w/ Source LLMs

For details on the evaluation, please refer to Section 4.1 of the paper.

| Model | MGSM-JA (acc &uarr;) | [lm-eval-harness](https://github.com/Stability-AI/lm-evaluation-harness/tree/jp-stable) (avg &uarr;) |
| :-- | --: | --: |
| [Shisa Gamma 7B v1](https://huggingface.co/augmxnt/shisa-gamma-7b-v1) | 9.6 | 66.1 |
| [WizardMath 7B V1.1](https://huggingface.co/WizardLM/WizardMath-7B-V1.1) | 18.4 | 60.1 |
| [Abel 7B 002](https://huggingface.co/GAIR/Abel-7B-002) | 30.0 | 56.5 |
| [Arithmo2 Mistral 7B](https://huggingface.co/upaya07/Arithmo2-Mistral-7B) | 24.0 | 56.4 |
| [EvoLLM-JP-A-v1-7B](https://huggingface.co/SakanaAI/EvoLLM-JP-A-v1-7B) | **52.4** | **69.0** |
| [EvoLLM-JP-v1-7B](https://huggingface.co/SakanaAI/EvoLLM-JP-v1-7B) | **52.0** | **70.5** |
| [EvoLLM-JP-v1-10B](https://huggingface.co/SakanaAI/EvoLLM-JP-v1-10B) | **55.6** | **66.2** |


### Comparing EvoVLM-JP w/ Existing VLMs

For details on the evaluation, please see Section 4.2 of the paper.


| Model | JA-VG-VQA-500 (ROUGE-L &uarr;) | JA-VLM-Bench-In-the-Wild (ROUGE-L &uarr;) |
| :-- | --: | --: |
| [LLaVA-1.6-Mistral-7B](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | 14.32 | 41.10 |
| [Japanese Stable VLM](https://huggingface.co/stabilityai/japanese-stable-vlm) | -<sup>*1</sup> | 40.50 |
| [Heron BLIP Japanese StableLM Base 7B llava-620k](https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1-llava-620k) | 14.51 | 33.26 |
| [EvoVLM-JP-v1-7B](https://huggingface.co/SakanaAI/EvoVLM-JP-v1-7B) | **19.70** | **51.25** |

* \*1: Japanese Stable VLM cannot be evaluated using the JA-VG-VQA-500 dataset because this model has used this dataset for training.




## Reproducing the Evaluation

### 1. Clone the Repo

```bash
git clone https://github.com/SakanaAI/evolutionary-model-merge.git
cd evolutionary-model-merge
```

### 2. Download fastext Model

We use fastext to detect language for evaluation. Please download `lid.176.ftz` from [this link](https://fasttext.cc/docs/en/language-identification.html) and place it in your current directory. If you place the file in a directory other than the current directory, specify the path to the file using the `LID176FTZ_PATH` environment variable.


### 3. Install Libraries

```bash
pip install -e .
```
We conducted our tests in the following environment: Python Version 3.10.12 and CUDA Version 12.3.
We cannot guarantee that it will work in other environments.

### 4. Run

To launch evaluation, run the following script with a certain config. All configs used for the paper are in `configs`.

```bash
python evaluate.py --config_path {path-to-config}
```


## Acknowledgement

We would like to thank the developers of the source models for their contributions and for making their work available. Our math evaluation code builds on the WizardMath repository, and we are grateful for their work.
