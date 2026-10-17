---
license: mit
language:
- en
library_name: transformers
pipeline_tag: text-classification
base_model: microsoft/deberta-v3-base
tags:
- text-classification
- content-moderation
- knowledge-distillation
- synthetic-data
- minecraft
- deberta-v3
- onnx
datasets:
- declip/Minecraft-Server-Chat
---

# Themis: DeBERTa-v3 Minecraft chat moderation

Themis is a small, fast moderation model for Minecraft chat. It is
[`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base),
fine-tuned on **synthetic labels produced by a decision model**, in this case
Perplexity's
[`perplexity-ai/pplx-decider-v1.1-27b`](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b).

The 27B decision model is accurate but expensive to run on every chat message.
We used it once, offline, to label 100k real chat messages, then distilled its
judgements into a 184M-parameter student. The student runs on a CPU at roughly
87 messages/second (ONNX, batch size 1, Ryzen 5 7600X).

For each message, the model returns two independent predictions:

| Task | Labels | Meaning |
| --- | --- | --- |
| `category` | `none`, `profanity`, `insult`, `slur`, `sexual`, `threat` | Which kind of offensive content the message contains, if any. |
| `action` | `allow`, `block_and_review` | Whether to let the message through or hold it for a human moderator, given a game setting with combat, banter and fictional violence. |

The action is predicted **independently** of the category. A message can be
`profanity` and still be `allow`. For example, "gg you absolute noob, I'll kill
you next round" is ordinary gameplay trash talk.

## How the training data was made

1. **Collect chat.** Public Minecraft chat from the
   [Minecraft Server Chat](https://huggingface.co/datasets/declip/Minecraft-Server-Chat)
   dataset and the 2025 [2b2t](https://2b2t.vc) chat archive, deduplicated.
2. **Sample for balance.** Most chat is harmless, so every message was
   pre-scored with
   [`KoalaAI/Text-Moderation`](https://huggingface.co/KoalaAI/Text-Moderation).
   A 100k sample was drawn using those scores to over-represent rare harmful
   content (about 50% "OK", the rest spread over hate, violence, sexual,
   self-harm and harassment). KoalaAI's labels are used only for sampling, not
   for training.
3. **Label pass 1: category.** `pplx-decider-v1.1-27b` was asked "Which single
   type of offensive content is this message, if any?", with a short criterion
   per label. We kept its full calibrated probability distribution, not just the
   top answer.
4. **Label pass 2: action.** The decider was run again on each message,
   told the pass 1 category and the game context (combat, friendly insults,
   fictional violence). It was asked whether to `allow` the message or
   `block_and_review` it, separating banter and in-game threats from targeted
   abuse, hate speech and credible real-world threats. Again we kept the full
   probability distribution.
5. **Distil.** DeBERTa-v3-base was trained on the message text alone, with no
   category hint. It has one 8-logit head: 6 category logits followed by 2
   action logits. The loss is the sum of two soft-target cross-entropies, one
   per task, against the teacher's probabilities. This teaches the student how
   confident the teacher was, not only which answer it picked.

Splits are grouped by normalised message text, so near-duplicates never cross
splits. They are also stratified by teacher category: 79,998 train / 10,002
validation / 9,999 test.

### Training setup

| Setting | Value |
| --- | --- |
| Base model | `microsoft/deberta-v3-base` @ `8ccc9b6f` |
| Teacher | `perplexity-ai/pplx-decider-v1.1-27b` @ `6195aa55`, unquantised BF16 |
| Max sequence length | 256 tokens |
| Epochs | 3 (best epoch chosen by validation loss) |
| Optimiser | AdamW, lr 2e-5, weight decay 0.01, linear schedule, 10% warmup |
| Batch size | 16 |
| Precision | FP32 |
| Hardware | 1× H100 (about 15 minutes) |

## Evaluation

These numbers measure **agreement with the teacher** on the held-out test
split. The "ground truth" is the argmax of the decider's probabilities.
They are not measured against human annotations.

| Test metric | Category | Action |
| --- | --- | --- |
| Agreement with teacher | 93.6% | 95.2% |
| Macro F1 | 0.840 | 0.920 |
| KL(teacher ‖ student) | 0.063 | 0.037 |

Per-class F1 on the test split:

| Category | F1 |
| --- | --- |
| `none` | 0.968 |
| `profanity` | 0.908 |
| `insult` | 0.720 |
| `slur` | 0.813 |
| `sexual` | 0.891 |
| `threat` | 0.742 |

| Action | F1 |
| --- | --- |
| `allow` | 0.971 |
| `block_and_review` | 0.869 |

## Usage

> [!IMPORTANT]
> The model outputs 8 logits. Apply softmax **separately** to `logits[:, :6]`
> (category) and `logits[:, 6:]` (action). A single softmax over all 8 logits
> gives meaningless probabilities. Load the tokenizer with `use_fast=False` to
> match training.

### Transformers (PyTorch)

```sh
pip install torch transformers sentencepiece protobuf
```

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "{{repo_id}}"
CATEGORIES = ["none", "profanity", "insult", "slur", "sexual", "threat"]
ACTIONS = ["allow", "block_and_review"]

tokenizer = AutoTokenizer.from_pretrained(REPO, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()


def moderate(texts: list[str]) -> list[dict]:
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256,
                       return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
    categories = logits[:, :6].softmax(dim=-1)  # independent softmax per task
    actions = logits[:, 6:].softmax(dim=-1)
    return [
        {
            "category": CATEGORIES[c.argmax()],
            "action": ACTIONS[a.argmax()],
            "category_probabilities": dict(zip(CATEGORIES, c.tolist())),
            "action_probabilities": dict(zip(ACTIONS, a.tolist())),
        }
        for c, a in zip(categories, actions)
    ]


for result in moderate(["gg wp everyone", "anyone want to trade diamonds?"]):
    print(result["category"], result["action"], result["action_probabilities"])
```

To trade recall for precision, use your own threshold on
`action_probabilities["block_and_review"]` instead of the argmax.

### ONNX Runtime (CPU, no PyTorch)

An FP32 ONNX export is in `onnx/`. It takes `input_ids` and `attention_mask`
(int64, `[batch, sequence]`) and returns `logits` (float32, `[batch, 8]`).
Tokenization happens outside the graph.

```sh
pip install onnxruntime transformers sentencepiece protobuf huggingface_hub
```

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO = "{{repo_id}}"
CATEGORIES = ["none", "profanity", "insult", "slur", "sexual", "threat"]
ACTIONS = ["allow", "block_and_review"]

tokenizer = AutoTokenizer.from_pretrained(REPO, use_fast=False)
session = ort.InferenceSession(hf_hub_download(REPO, "onnx/model.onnx"),
                               providers=["CPUExecutionProvider"])


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


texts = ["gg wp everyone", "anyone want to trade diamonds?"]
encoded = tokenizer(texts, padding=True, truncation=True, max_length=256,
                    return_tensors="np", return_token_type_ids=False)
(logits,) = session.run(["logits"], {
    "input_ids": encoded["input_ids"].astype(np.int64),
    "attention_mask": encoded["attention_mask"].astype(np.int64),
})
for text, c, a in zip(texts, softmax(logits[:, :6]), softmax(logits[:, 6:])):
    print(text, CATEGORIES[c.argmax()], ACTIONS[a.argmax()], float(a[1]))
```

`onnx/inference_config.json` records the logit groups, label order, token
limit and tokenizer settings. Use it to port the model to other runtimes.
The [Themis Java library](https://github.com/utfunderscore/themis/tree/main/java)
runs this bundle from Java 21 with ONNX Runtime.

## Limitations

- **Teacher-bounded.** The student learns the decider's judgement, including
  its mistakes and biases. The metrics above show fidelity to the teacher, not
  real-world accuracy.
- **Domain.** Trained on Minecraft server chat: short, informal, mostly
  English, full of game slang. Expect weaker results on other platforms,
  long-form text or other languages.
- **Weaker classes.** `insult` and `threat` are the hardest categories
  (F1 ≈ 0.72–0.74 against the teacher). Here, banter and real harassment are
  hard to tell apart.
- **No context.** Each message is classified alone, with no conversation
  history, sender or server rules.
- **Human in the loop.** `block_and_review` is meant to send a message to a
  human moderator, not to punish anyone automatically.

## Source

Training, labelling and export code: <https://github.com/utfunderscore/themis>.
