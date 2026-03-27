"""Fast training script - only train.json"""
import json
from model import POSTagger

print("=" * 50)
print("Fast POS Tagger Training")
print("=" * 50)

# Load only train.json (consistent tagset)
print("\n[1] Loading train.json...")
Text, Label = [], []
with open("train.json", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        Text.append(data["words"])
        Label.append(data["labels"])

print(f"    Total: {len(Text)} sentences, {sum(len(s) for s in Text)} tokens")

# Train
print("\n[2] Training...")
tagger = POSTagger()
tagger.fit(Text, Label)

# Save
print("\n[3] Saving...")
tagger.save()

# Validate
print("\n[4] Validation...")
preds = tagger.predict(Text[:1000])
correct = sum(1 for p, g in zip(preds, Label[:1000]) for a, b in zip(p, g) if a == b)
total = sum(len(g) for g in Label[:1000])
print(f"    Accuracy: {correct/total*100:.2f}%")

print("\nDone! Model saved to model.mdl")