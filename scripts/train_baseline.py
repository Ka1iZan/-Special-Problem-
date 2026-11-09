"""Ordinary XLM-R baseline. GPU execution is a separate validation gate.

No prototype alignment/replay here. No test evaluation unless explicitly requested.
Long records fail preflight rather than silently losing entities to truncation.
"""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import _bootstrap  # noqa: F401
from spner.data.io import read_jsonl
from spner.data.tokenize import LABELS, LABEL_TO_ID, align_first_subwords
from spner.metrics import score_sentence, aggregate_scores
from spner.utils import file_sha256, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "tiny", "baseline"], default="smoke")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new run directory; resume is not implemented in this pilot runner")
    if args.evaluate_test and args.mode != "baseline":
        parser.error("Smoke/tiny diagnostics cannot evaluate test data")
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup

    config = json.loads(args.config.read_text(encoding="utf-8"))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    if config["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("Select a Kaggle GPU accelerator before running this training gate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = read_jsonl(args.data / "train.jsonl")
    dev = read_jsonl(args.data / "dev.jsonl")
    if any(r.split != "train" for r in train) or any(r.split != "dev" for r in dev):
        raise ValueError("Incorrect train/dev split metadata")
    if len({r.language for r in train + dev}) != 1:
        raise ValueError("This monolingual gate requires one language")
    if args.mode in {"smoke", "tiny"}:
        train = random.Random(config["seed"]).sample(train, min(32, len(train)))
        dev = train if args.mode == "tiny" else dev[:32]
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=config["model_revision"], use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer with word_ids() required")

    def encode(records):
        encoded = []
        too_long = []
        for record in records:
            item = tokenizer(record.tokens, is_split_into_words=True, truncation=False)
            if len(item["input_ids"]) > config["max_length"]:
                too_long.append({"example_id": record.example_id, "subwords": len(item["input_ids"])})
                continue
            aligned = align_first_subwords(item.word_ids(), record.labels_coarse)
            if all(label == -100 for label in aligned["labels"]):
                raise ValueError(f"No supervised positions in {record.example_id}")
            encoded.append({"features": dict(item), "labels": aligned["labels"],
                            "positions": aligned["first_subword_positions"], "record": record})
        if too_long:
            raise ValueError(f"{len(too_long)} records exceed max_length; implement explicit windows before training. First cases: {too_long[:5]}")
        return encoded

    train_encoded, dev_encoded = encode(train), encode(dev)

    def collate(items):
        batch = tokenizer.pad([item["features"] for item in items], padding=True, return_tensors="pt")
        width = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([item["labels"] + [-100] * (width - len(item["labels"])) for item in items])
        return batch, items

    loader = DataLoader(train_encoded, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]), collate_fn=collate)
    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name"], revision=config["model_revision"], num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID).to(device)
    args.out.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "transformers": importlib.metadata.version("transformers"),
                   "git_commit": commit.stdout.strip() if commit.returncode == 0 else "UNCOMMITTED",
                   "git_dirty": bool(dirty.stdout.strip()),
                   "resolved_model_revision": getattr(model.config, "_commit_hash", None),
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    save_json(args.out / "environment.json", environment)
    save_json(args.out / "config.resolved.json", {**config, "mode": args.mode, "evaluate_test": args.evaluate_test})
    save_json(args.out / "data_manifest.json", {"processed_manifest_sha256": file_sha256(args.data / "manifest.json"),
              "train_ids": [r.example_id for r in train], "dev_ids": [r.example_id for r in dev],
              "train_file_sha256": file_sha256(args.data / "train.jsonl"),
              "dev_file_sha256": file_sha256(args.data / "dev.jsonl")})
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (args.out / "environment-lock.txt").write_text(frozen.stdout, encoding="utf-8")
    epochs = 1 if args.mode == "smoke" else (50 if args.mode == "tiny" else config["epochs"])
    accumulation = config["gradient_accumulation_steps"]
    total_steps = 1 if args.mode == "smoke" else epochs * math.ceil(len(loader) / accumulation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * config["warmup_ratio"]), total_steps)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)

    def precision():
        return torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else nullcontext()

    def evaluate(encoded, name):
        model.eval()
        scores, predictions = [], []
        with torch.no_grad():
            for batch, items in DataLoader(encoded, batch_size=config["batch_size"], collate_fn=collate):
                with precision():
                    logits = model(**{k: v.to(device) for k, v in batch.items() if k != "labels"}).logits
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite evaluation logits")
                ids = logits.argmax(-1).cpu().tolist()
                for row, item in zip(ids, items):
                    labels = [LABELS[row[position]] for position in item["positions"]]
                    record = item["record"]
                    scores.append(score_sentence(record.tokens, record.labels_coarse, labels,
                                                  ignore_policy=config["ignore_policy"]))
                    predictions.append({"example_id": record.example_id, "labels": labels})
        metrics = aggregate_scores(scores)
        (args.out / f"{name}.predictions.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in predictions) + "\n", encoding="utf-8")
        return metrics

    best, stale, step = -1.0, 0, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch_index, (batch, _) in enumerate(loader):
            group_start = (batch_index // accumulation) * accumulation
            divisor = min(accumulation, len(loader) - group_start)
            with precision():
                loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite NER loss")
            losses.append(float(loss.detach()))
            scaler.scale(loss / divisor).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                if step == 0 and float(norm) <= 0:
                    raise RuntimeError("No training gradient")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if args.mode == "smoke":
                    break
        metrics = evaluate(dev_encoded, "latest_dev" if args.mode != "tiny" else "latest_tiny_train")
        f1 = metrics["all"]["f1"]
        if f1 is None:
            raise ValueError("No scored entities in evaluation gate")
        entry = {"epoch": epoch + 1, "optimizer_steps": step, "mean_ner_loss": sum(losses) / len(losses),
                 "evaluation_f1": f1, "mode": args.mode}
        with (args.out / "train_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        if f1 > best:
            best, stale = f1, 0
            model.save_pretrained(args.out / "checkpoint-best")
            tokenizer.save_pretrained(args.out / "checkpoint-best")
            (args.out / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            latest = "latest_tiny_train" if args.mode == "tiny" else "latest_dev"
            shutil.copyfile(args.out / f"{latest}.predictions.jsonl", args.out / "best_evaluation.predictions.jsonl")
            (args.out / "best_selection.json").write_text(json.dumps({"epoch": epoch + 1,
                "optimizer_steps": step, "evaluation_split": "train" if args.mode == "tiny" else "dev"}), encoding="utf-8")
        else:
            stale += 1
        if args.mode == "tiny" and best >= .95:
            break
        if args.mode == "baseline" and stale >= config["patience"]:
            break
    if args.evaluate_test:
        model = AutoModelForTokenClassification.from_pretrained(args.out / "checkpoint-best").to(device)
        test = read_jsonl(args.data / "test.jsonl")
        if any(r.split != "test" or r.language != train[0].language for r in test):
            raise ValueError("Incorrect test language/split")
        save_json(args.out / "test_manifest.json", {"test_file_sha256": file_sha256(args.data / "test.jsonl"),
                  "test_ids": [r.example_id for r in test]})
        save_json(args.out / "test_metrics.json", evaluate(encode(test), "test"))
    save_json(args.out / "completion.json", {"mode": args.mode, "best_evaluation_f1": best,
              "optimizer_steps": step, "tiny_overfit_passed": best >= .95 if args.mode == "tiny" else None,
              "is_research_result": args.mode == "baseline", "test_evaluated": args.evaluate_test})
    print(f"Completed {args.mode} gate: {args.out}")
    if args.mode == "tiny" and best < .95:
        raise RuntimeError("Tiny overfit gate did not reach 0.95 training entity F1; inspect before scaling up")


if __name__ == "__main__":
    main()
