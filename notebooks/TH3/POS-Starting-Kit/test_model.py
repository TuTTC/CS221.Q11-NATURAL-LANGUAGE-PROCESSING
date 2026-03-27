"""
Test script for the POS Tagger model (model.mdl)
This script tests the trained CRF model by:
1. Loading the model
2. Running predictions on test sentences
3. Calculating accuracy metrics
"""

import json
from model import POSTagger
from sklearn_crfsuite import metrics

def load_data(filepath):
    """Load data from JSONL file"""
    texts = []
    labels = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            texts.append(data["words"])
            labels.append(data["labels"])
    return texts, labels

def test_single_sentence(tagger, sentence):
    """Test a single sentence and print predictions"""
    predictions = tagger.predict([sentence])
    print("\n=== Single Sentence Test ===")
    print(f"Sentence: {sentence}")
    print("\nWord-by-word predictions:")
    for word, tag in zip(sentence, predictions[0]):
        print(f"  {word:20s} -> {tag}")

def test_on_dataset(tagger, texts, labels, num_samples=100):
    """Test on a portion of the dataset"""
    # Use first num_samples for testing
    test_texts = texts[:num_samples]
    test_labels = labels[:num_samples]
    
    # Get predictions
    predictions = tagger.predict(test_texts)
    
    # Calculate metrics
    print(f"\n=== Dataset Test (first {num_samples} samples) ===")
    
    # Flatten for accuracy calculation
    y_true = [tag for sent in test_labels for tag in sent]
    y_pred = [tag for sent in predictions for tag in sent]
    
    # Calculate accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    accuracy = correct / total * 100
    
    print(f"Total tokens: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Get unique labels for detailed report
    unique_labels = list(set(y_true))
    unique_labels.sort()
    
    print("\n=== Classification Report ===")
    print(metrics.flat_classification_report(
        test_labels, predictions, labels=unique_labels, digits=3
    ))
    
    return accuracy

def test_custom_sentences(tagger):
    """Test with custom example sentences"""
    print("\n=== Custom Sentence Tests ===")
    
    test_sentences = [
        ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."],
        ["I", "love", "programming", "in", "Python", "."],
        ["Natural", "language", "processing", "is", "fascinating", "."],
        ["Apple", "Computer", "Inc.", "announced", "new", "products", "yesterday", "."],
        ["The", "stock", "market", "fell", "sharply", "today", "."]
    ]
    
    for sentence in test_sentences:
        predictions = tagger.predict([sentence])
        print(f"\nSentence: {' '.join(sentence)}")
        print("Tags:", ' '.join(predictions[0]))
        print("Word-Tag pairs:")
        for word, tag in zip(sentence, predictions[0]):
            print(f"  {word:20s} -> {tag}")

def main():
    print("=" * 60)
    print("POS Tagger Model Test")
    print("=" * 60)
    
    # Initialize tagger and load model
    print("\n[1] Loading model from model.mdl...")
    tagger = POSTagger()
    try:
        tagger.load("model.mdl")
        print("    Model loaded successfully!")
    except FileNotFoundError:
        print("    ERROR: model.mdl not found!")
        print("    Please train the model first by running: python train.py")
        return
    except Exception as e:
        print(f"    ERROR loading model: {e}")
        return
    
    # Load training data for testing
    print("\n[2] Loading test data from train.json...")
    try:
        texts, labels = load_data("train.json")
        print(f"    Loaded {len(texts)} sentences")
    except FileNotFoundError:
        print("    WARNING: train.json not found!")
        texts, labels = [], []
    
    # Test with single sentence
    print("\n[3] Testing single sentence prediction...")
    sample_sentence = ["The", "company", "reported", "strong", "earnings", "."]
    test_single_sentence(tagger, sample_sentence)
    
    # Test on dataset
    if texts and labels:
        print("\n[4] Testing on dataset...")
        accuracy = test_on_dataset(tagger, texts, labels, num_samples=500)
    
    # Test custom sentences
    print("\n[5] Testing custom sentences...")
    test_custom_sentences(tagger)
    
    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
