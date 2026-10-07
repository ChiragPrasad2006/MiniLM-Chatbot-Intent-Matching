"""Train a CLINC150 intent model with validation-only OOS calibration."""

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


DATASET_PATH = Path("./clinc150/clinc150_uci/data_oos_plus.json")
OOS_AUGMENTATION_PATH = Path("./data/oos_augmentation.json")
CHECKPOINT_DIR = "./models/minilm_intent_matching_improved_checkpoints"
MODEL_SAVE_PATH = Path("./models/minilm_intent_matching_improved/final")
MODEL_ID = "sentence-transformers/all-MiniLM-L12-v2"
MAX_LENGTH = 64
SEED = 42
OOS_LOSS_WEIGHT = 2.0
MAX_KNOWN_REJECTION_RATE = 0.05


def normalize_text(text):
    """Normalize text for exact-duplicate checks across dataset splits."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def load_splits():
    with DATASET_PATH.open(encoding="utf-8") as data_file:
        raw = json.load(data_file)

    train = list(raw["train"]) + list(raw["oos_train"])
    validation = list(raw["val"]) + list(raw["oos_val"])
    test = list(raw["test"]) + list(raw["oos_test"])

    # Keep exact held-out utterances out of training, including the few
    # conflicting duplicate labels present in the source files.
    heldout_texts = {normalize_text(text) for text, _ in validation + test}
    original_count = len(train)
    train = [(text, label) for text, label in train if normalize_text(text) not in heldout_texts]
    print(f"Removed {original_count - len(train)} exact train/held-out text overlaps.")

    with OOS_AUGMENTATION_PATH.open(encoding="utf-8") as data_file:
        curated_oos = json.load(data_file)
    if not isinstance(curated_oos, list) or not all(isinstance(text, str) for text in curated_oos):
        raise ValueError(f"{OOS_AUGMENTATION_PATH} must contain a JSON list of strings.")

    # Keep a deterministic slice of the curated OOS examples out of training.
    # These join the original validation split and are used only for selection
    # and calibration. The official test split remains untouched.
    rng = random.Random(SEED)
    rng.shuffle(curated_oos)
    calibration_count = max(1, round(len(curated_oos) * 0.20))
    curated_validation = [(text, "oos") for text in curated_oos[:calibration_count]]
    curated_training = [(text, "oos") for text in curated_oos[calibration_count:]]

    known_heldout = {normalize_text(text) for text, _ in validation + test}
    trained_texts = {normalize_text(text) for text, _ in train}
    for text, _ in curated_training:
        normalized = normalize_text(text)
        if normalized in known_heldout or normalized in trained_texts:
            raise ValueError(f"OOS augmentation overlaps another split: {text!r}")
        trained_texts.add(normalized)
    for text, _ in curated_validation:
        normalized = normalize_text(text)
        if normalized in trained_texts or normalized in known_heldout:
            raise ValueError(f"OOS calibration example overlaps another split: {text!r}")
        known_heldout.add(normalized)

    train.extend(curated_training)
    validation.extend(curated_validation)

    print(
        "Split sizes: "
        f"train={len(train)} (OOS={sum(y == 'oos' for _, y in train)}), "
        f"validation={len(validation)} (OOS={sum(y == 'oos' for _, y in validation)}), "
        f"test={len(test)} (OOS={sum(y == 'oos' for _, y in test)})."
    )
    return train, validation, test


def make_dataset(rows, label2id):
    frame = pd.DataFrame(rows, columns=["text", "intent"])
    frame["labels"] = frame["intent"].map(label2id)
    if frame["labels"].isna().any():
        raise ValueError("Found an intent label that is absent from the training split.")
    return Dataset.from_pandas(frame, preserve_index=False)


def metrics_from_logits(logits, labels, oos_id):
    predictions = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    known_mask = labels != oos_id
    predicted_oos = predictions == oos_id
    true_oos = labels == oos_id
    known_rejected = known_mask & predicted_oos

    oos_tp = int(np.sum(true_oos & predicted_oos))
    oos_fp = int(np.sum(known_rejected))
    oos_fn = int(np.sum(true_oos & ~predicted_oos))
    oos_precision = oos_tp / max(1, oos_tp + oos_fp)
    oos_recall = oos_tp / max(1, oos_tp + oos_fn)
    oos_f1 = 2 * oos_precision * oos_recall / max(1e-12, oos_precision + oos_recall)

    return {
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "known_accuracy": float(np.mean(predictions[known_mask] == labels[known_mask])),
        "oos_precision": float(oos_precision),
        "oos_recall": float(oos_recall),
        "oos_f1": float(oos_f1),
        "known_false_oos_rate": float(np.mean(predicted_oos[known_mask])),
    }


def fit_temperature(logits, labels):
    """Fit one scalar temperature on validation logits, minimizing NLL."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def select_oos_threshold(probabilities, labels, oos_id):
    """Maximize OOS recall subject to a validation known-rejection limit."""
    predictions = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)
    labels = np.asarray(labels)
    known_mask = labels != oos_id
    oos_mask = ~known_mask
    candidates = np.unique(np.concatenate(([0.0], confidence, [1.0 + 1e-9])))
    valid = []
    for threshold in candidates:
        final_predictions = predictions.copy()
        final_predictions[confidence < threshold] = oos_id
        known_rejection_rate = float(np.mean(final_predictions[known_mask] == oos_id))
        oos_recall = float(np.mean(final_predictions[oos_mask] == oos_id))
        if known_rejection_rate <= MAX_KNOWN_REJECTION_RATE:
            valid.append((oos_recall, -known_rejection_rate, -float(threshold), float(threshold)))
    return max(valid)[-1] if valid else 0.0


def threshold_metrics(probabilities, labels, oos_id, threshold):
    predictions = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)
    predictions[confidence < threshold] = oos_id
    return metrics_from_logits(np.eye(probabilities.shape[1])[predictions], labels, oos_id)


class OOSWeightedTrainer(Trainer):
    """Apply a modest extra loss weight to the underrepresented OOS class."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = F.cross_entropy(outputs.logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def save_per_intent_report(logits, labels, id2label, destination):
    predictions = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    rows = []
    for label_id, label in sorted(id2label.items()):
        mask = labels == int(label_id)
        if not np.any(mask):
            continue
        rows.append({
            "intent": label,
            "support": int(mask.sum()),
            "accuracy": float(np.mean(predictions[mask] == labels[mask])),
        })
    pd.DataFrame(rows).sort_values("accuracy").to_csv(destination, index=False)


def save_confusion_pairs(logits, labels, id2label, destination):
    predictions = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    pairs = {}
    for true_id, pred_id in zip(labels, predictions):
        if true_id == pred_id:
            continue
        pairs[(int(true_id), int(pred_id))] = pairs.get((int(true_id), int(pred_id)), 0) + 1
    rows = [
        {
            "true_intent": id2label[true_id],
            "predicted_intent": id2label[pred_id],
            "count": count,
        }
        for (true_id, pred_id), count in pairs.items()
    ]
    pd.DataFrame(rows, columns=["true_intent", "predicted_intent", "count"]).sort_values(
        "count", ascending=False
    ).to_csv(destination, index=False)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_rows, validation_rows, test_rows = load_splits()
    unique_intents = sorted({label for _, label in train_rows})
    label2id = {label: idx for idx, label in enumerate(unique_intents)}
    id2label = {idx: label for label, idx in label2id.items()}
    oos_id = label2id["oos"]
    print(f"Loaded {len(unique_intents)} unique intent classes (including 'oos' fallback).")

    dataset = DatasetDict({
        "train": make_dataset(train_rows, label2id),
        "validation": make_dataset(validation_rows, label2id),
        "test": make_dataset(test_rows, label2id),
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(unique_intents),
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = DatasetDict({
        split: split_data.map(tokenize_function, batched=True)
        for split, split_data in dataset.items()
    })

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        return metrics_from_logits(logits, labels, oos_id)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=12,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=max(1, round((len(tokenized["train"]) / 64) * 12 * 0.1)),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=0,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )

    class_weights = torch.ones(len(unique_intents), dtype=torch.float32)
    class_weights[oos_id] = OOS_LOSS_WEIGHT
    trainer = OOSWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training with macro-F1 checkpoint selection and early stopping...")
    trainer.train()

    # Calibrate only after validation selected the checkpoint. Never use test
    # logits for temperature or threshold selection.
    validation_output = trainer.predict(tokenized["validation"], metric_key_prefix="validation")
    validation_logits = validation_output.predictions
    validation_labels = validation_output.label_ids
    temperature = fit_temperature(validation_logits, validation_labels)
    calibrated_validation = torch.softmax(
        torch.as_tensor(validation_logits, dtype=torch.float64) / temperature, dim=-1
    ).cpu().numpy()
    threshold = select_oos_threshold(calibrated_validation, validation_labels, oos_id)
    calibrated_metrics = threshold_metrics(
        calibrated_validation, validation_labels, oos_id, threshold
    )

    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_SAVE_PATH))
    tokenizer.save_pretrained(str(MODEL_SAVE_PATH))
    calibration = {
        "temperature": temperature,
        "oos_threshold": threshold,
        "max_length": MAX_LENGTH,
        "threshold_selection": "max_oos_recall_at_max_5pct_known_rejection_on_validation",
        "validation_metrics_after_threshold": calibrated_metrics,
        "validation_metrics_raw_argmax": validation_output.metrics,
    }
    with (MODEL_SAVE_PATH / "confidence_calibration.json").open("w", encoding="utf-8") as out:
        json.dump(calibration, out, indent=2)
    save_per_intent_report(
        validation_logits,
        validation_labels,
        id2label,
        MODEL_SAVE_PATH / "validation_per_intent_accuracy.csv",
    )
    save_confusion_pairs(
        validation_logits,
        validation_labels,
        id2label,
        MODEL_SAVE_PATH / "validation_confusion_pairs.csv",
    )

    print("\nValidation calibration:", json.dumps(calibration, indent=2))
    print(f"Per-intent validation accuracy saved to {MODEL_SAVE_PATH / 'validation_per_intent_accuracy.csv'}")
    print(f"Validation confusion pairs saved to {MODEL_SAVE_PATH / 'validation_confusion_pairs.csv'}")

    # Official test is evaluated once, after model selection and calibration.
    test_output = trainer.predict(tokenized["test"], metric_key_prefix="test")
    test_probabilities = torch.softmax(
        torch.as_tensor(test_output.predictions, dtype=torch.float64) / temperature, dim=-1
    ).cpu().numpy()
    test_calibrated_metrics = threshold_metrics(
        test_probabilities, test_output.label_ids, oos_id, threshold
    )
    print("\nOfficial test metrics (raw argmax):", json.dumps(test_output.metrics, indent=2))
    print("Official test metrics (validation-calibrated OOS threshold):",
          json.dumps(test_calibrated_metrics, indent=2))


if __name__ == "__main__":
    main()
