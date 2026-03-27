"""
BERT POS Tagger Training Script for Kaggle
Upload train.json to Kaggle and run this notebook
"""

# Install dependencies (run in first cell)
# !pip install transformers datasets accelerate -q

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast, 
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.model_selection import train_test_split
import pickle

# ============ LOAD DATA ============
print("Loading data...")
texts, labels = [], []
with open("train.json", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["words"])
        labels.append(data["labels"])

print(f"Loaded {len(texts)} sentences")

# Build label mapping
all_labels = sorted(set(tag for tags in labels for tag in tags))
label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(all_labels)
print(f"Number of labels: {num_labels}")

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# ============ TOKENIZATION ============
print("Tokenizing...")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_and_align_labels(words, tags):
    """Tokenize and align labels with subwords"""
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )
    
    word_ids = encoding.word_ids()
    label_ids = []
    prev_word_id = None
    
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)  # Special tokens
        elif word_id != prev_word_id:
            label_ids.append(label2id[tags[word_id]])
        else:
            label_ids.append(-100)  # Subword
        prev_word_id = word_id
    
    return {
        'input_ids': encoding['input_ids'].squeeze(),
        'attention_mask': encoding['attention_mask'].squeeze(),
        'labels': torch.tensor(label_ids)
    }

class POSDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return tokenize_and_align_labels(self.texts[idx], self.labels[idx])

train_dataset = POSDataset(train_texts, train_labels)
val_dataset = POSDataset(val_texts, val_labels)

# ============ MODEL ============
print("Loading BERT model...")
model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# ============ TRAINING ============
print("Setting up training...")

training_args = TrainingArguments(
    output_dir='./pos_bert',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    fp16=True,  # Enable mixed precision for faster training
)

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)
    
    # Flatten and remove ignored tokens
    true_labels = []
    true_preds = []
    for p, l in zip(predictions, labels):
        for pred_id, label_id in zip(p, l):
            if label_id != -100:
                true_labels.append(label_id)
                true_preds.append(pred_id)
    
    accuracy = sum(1 for t, p in zip(true_labels, true_preds) if t == p) / len(true_labels)
    return {'accuracy': accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Training...")
trainer.train()

# ============ SAVE MODEL ============
print("Saving model...")
model.save_pretrained('./pos_bert_final')
tokenizer.save_pretrained('./pos_bert_final')

# Save label mappings
with open('./pos_bert_final/label_mapping.pkl', 'wb') as f:
    pickle.dump({'label2id': label2id, 'id2label': id2label}, f)

print("Done! Download the pos_bert_final folder")

# ============ EVALUATE ============
print("\nEvaluating on validation set...")
results = trainer.evaluate()
print(f"Validation accuracy: {results['eval_accuracy']:.4f}")
