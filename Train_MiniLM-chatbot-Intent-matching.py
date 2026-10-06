import json
import torch
import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# 1. Load dataset with Out-of-Scope (OOS) data (CLINC150 data_oos_plus)
dataset_path = "./clinc150/clinc150_uci/data_oos_plus.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Merge in-domain and out-of-scope (OOS) samples for each split:
# - train: 15,000 in-domain + 250 oos = 15,250 samples
# - val:    3,000 in-domain + 100 oos =  3,100 samples
# - test:   4,500 in-domain + 1,000 oos = 5,500 samples
train_data = raw_data["train"] + raw_data["oos_train"]
val_data = raw_data["val"] + raw_data["oos_val"]
test_data = raw_data["test"] + raw_data["oos_test"]

# Extract 151 intent labels (150 in-domain + 'oos' fallback class)
unique_intents = sorted(list(set(x[1] for x in train_data)))
label2id = {label: idx for idx, label in enumerate(unique_intents)}
id2label = {idx: label for idx, label in enumerate(unique_intents)}
num_labels = len(unique_intents)
print(f"Loaded {num_labels} unique intent classes (including 'oos' fallback).")

# Helper to convert list of [text, intent] pairs to HF Dataset with numeric label
def make_split(split_data):
    df = pd.DataFrame(split_data, columns=["text", "intent"])
    df["label"] = df["intent"].map(label2id)
    return Dataset.from_pandas(df)

# Create Hugging Face DatasetDict
dataset = DatasetDict({
    "train": make_split(train_data),       
    "validation": make_split(val_data),       
    "test": make_split(test_data),            
})
print("Dataset structure:\n", dataset)

# 2. Model & Tokenizer
model_id = "sentence-transformers/all-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# max_length=64 covers 100% of samples without wasteful padding
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64)

tokenized_train_data = dataset["train"].map(tokenize_function, batched=True)
tokenized_val_data = dataset["validation"].map(tokenize_function, batched=True)
tokenized_test_data = dataset["test"].map(tokenize_function, batched=True)

# 3. Evaluation Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1": f1}

# Hardware check for mixed precision
use_fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# 4. Training Arguments tailored for CLINC150 (151 classes) & MiniLM
# Math:
#   Train samples = 15,250
#   Batch size = 64 -> ~239 steps per epoch
#   Epochs = 5 -> Total steps = ~1,195
#   Warmup (10%) = ~120 steps
training_args = TrainingArguments(
    output_dir="./models/minilm_intent_matching_checkpoints",
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=120,                          # 10% warmup (~120 steps for 5 epochs)
    logging_steps=50,
    eval_strategy="epoch",                     
    save_strategy="epoch",                     
    save_total_limit=2,                        
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=use_fp16,
    bf16=use_bf16,
    dataloader_num_workers=0,                  
    report_to="none"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_val_data,           # Use validation split during training & model selection
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    processing_class=tokenizer
)

# 6. Train & Evaluate
print("Starting training...")
trainer.train()

print("\n--- Evaluating on Validation Set ---")
val_results = trainer.evaluate(eval_dataset=tokenized_val_data)
print("Validation Results:", val_results)

print("\n--- Evaluating on Unseen Test Set (including 1,000 OOS queries) ---")
test_results = trainer.evaluate(eval_dataset=tokenized_test_data)
print("Test Results:", test_results)

# 7. Save final model and tokenizer
model_save_path = "./models/minilm_intent_matching/final"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\nModel and tokenizer saved successfully to: {model_save_path}")