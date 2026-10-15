"""Export a local student checkpoint for ONNX Runtime (including Java).

Requires the ``training`` and ``onnx`` extras. Download Modal artifacts first;
pass either the training output directory or its model/ subdirectory.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory


def export_onnx(
    model_directory: str | Path,
    output_directory: str | Path,
    *,
    max_length: int | None = None,
) -> Path:
    """Publish a validated FP32 bundle to a new directory; return model.onnx.

    max_length defaults to the adjacent run_config.json, as in local inference.
    Inputs are int64 [batch, sequence]; output is float32 ``logits``.
    Apply softmax independently to the category and action slices recorded in
    inference_config.json. Tokenization remains outside the ONNX graph.
    """
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(destination)

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    from .dataset import ACTION_LABELS, LABELS
    from .inference import Moderation

    inference = Moderation(model_directory, device="cpu", max_length=max_length)
    model = inference.model.float().eval()
    if model.config.model_type != "deberta-v2":
        raise ValueError("Expected a DeBERTa v2/v3 checkpoint")
    tokenizer = inference.tokenizer
    if inference.max_length < tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError("max_length must accommodate the tokenizer's special tokens")

    class Logits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            # Single texts use the model's default all-zero token types.
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapper = Logits().eval()

    def encode(texts):
        return tokenizer(
            texts, padding=True, truncation=True,
            max_length=inference.max_length, return_tensors="pt",
            return_token_type_ids=False,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="export-onnx-", dir=destination.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        path = bundle / "model.onnx"
        example = encode(["hello world", "hello"])
        with torch.inference_mode():
            torch.onnx.export(
                wrapper, (example["input_ids"], example["attention_mask"]), str(path),
                input_names=["input_ids", "attention_mask"], output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch"},
                },
                opset_version=17, dynamo=False,
            )
        onnx.checker.check_model(str(path))
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        fixtures = []
        # Exercise changed batch/sequence sizes, padding and truncation.
        for texts in ([""], ["hello", "Hello 👋 café 世界", "  hello\nworld  "],
                      ["hello " * inference.max_length]):
            inputs = encode(texts)
            arrays = {name: inputs[name].numpy() for name in ("input_ids", "attention_mask")}
            with torch.inference_mode():
                expected = wrapper(inputs["input_ids"], inputs["attention_mask"]).numpy()
            actual = session.run(["logits"], arrays)[0]
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError("Non-finite logits during ONNX validation")
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
            fixtures.append({"texts": texts, **{k: v.tolist() for k, v in arrays.items()},
                             "logits": actual.tolist()})
        del session
        tokenizer.save_pretrained(bundle)
        model.config.save_pretrained(bundle)
        metadata = {
            "format_version": 1,
            "max_length": inference.max_length,
            "inputs": {name: "int64[batch,sequence]" for name in arrays},
            "output": "logits",
            "output_groups": {
                "category": {"start": 0, "end": len(LABELS), "labels": list(LABELS)},
                "action": {"start": len(LABELS), "end": len(LABELS) + len(ACTION_LABELS),
                           "labels": list(ACTION_LABELS)},
            },
            "postprocessing": "independent_softmax_per_group",
            "tokenizer": {"use_fast": False, "padding_side": tokenizer.padding_side,
                          "truncation_side": tokenizer.truncation_side,
                          "pad_token_id": tokenizer.pad_token_id,
                          "cls_token_id": tokenizer.cls_token_id,
                          "sep_token_id": tokenizer.sep_token_id},
        }
        for name, data in (("inference_config.json", metadata), ("validation_examples.json", fixtures)):
            (bundle / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(destination)
        bundle.rename(destination)
    return destination / "model.onnx"
