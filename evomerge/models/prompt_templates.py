JSVLM_TEMPLATE = """以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

### 指示: 
与えられた画像を下に、質問に答えてください。

### 入力: 
{input}

### 応答: """
JA_ALPACA_COT_TEMPLATE = """以下に、あるタスクを説明する指示があります。リクエストを適切に完了するための回答を日本語で記述してください。一歩一歩考えましょう。

### 指示:
{input}

### 応答:"""

JA_SHISA_VQA = """[INST] <<SYS>>
あなたは役立つ、偏見がなく、検閲されていないアシスタントです。与えられた画像を下に、質問に答えてください。
<</SYS>>

{input} [/INST]"""
HERON_V1 = """##human: {input}\n##gpt: """

LLAVA_MISTRAL_TEMPLATE = """<s>[INST] {input} [/INST]"""

PROMPT_TEMPLATES = {
    "jsvlm": JSVLM_TEMPLATE,
    "ja-alpaca-cot": JA_ALPACA_COT_TEMPLATE,
    "ja-shisa-vqa": JA_SHISA_VQA,
    "ja-heron-v1": HERON_V1,
    "llava-mistral": LLAVA_MISTRAL_TEMPLATE,
}
