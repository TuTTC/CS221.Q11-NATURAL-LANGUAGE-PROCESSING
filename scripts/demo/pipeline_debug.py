"""
Detailed Pipeline Debug Script
Shows tensor shapes and values at each step of BiLSTM-CRF + BiLSTM-Attention pipeline
"""

import torch
import torch.nn as nn
from torchcrf import CRF
import pickle
from typing import List, Dict

# ============================================================================
# 1. MODEL DEFINITIONS
# ============================================================================

class BiLSTM_CRF(nn.Module):
    """BiLSTM-CRF for Aspect-Opinion extraction"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, x, tags=None):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        emissions = self.fc(lstm_out)
        if tags is not None:
            loss = -self.crf(emissions, tags)
            return loss
        else:
            return self.crf.decode(emissions)
    
    def forward_debug(self, x):
        """Debug forward pass - returns intermediate tensors"""
        emb = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(emb)
        emissions = self.fc(lstm_out)
        decoded = self.crf.decode(emissions)
        return {
            'embedding': emb,
            'lstm_output': lstm_out,
            'emissions': emissions,
            'decoded': decoded,
            'hidden_state': h_n,
            'cell_state': c_n
        }


class BiLSTM_Attention(nn.Module):
    """BiLSTM with Attention for Sentiment Classification"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        h, _ = self.lstm(emb)
        weights = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        rep = torch.sum(h * weights.unsqueeze(-1), dim=1)
        return self.fc(rep)
    
    def forward_debug(self, x):
        """Debug forward pass - returns intermediate tensors"""
        emb = self.embedding(x)
        h, (h_n, c_n) = self.lstm(emb)
        attn_scores = self.attn(h).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)
        logits = self.fc(context)
        return {
            'embedding': emb,
            'lstm_output': h,
            'attention_scores': attn_scores,
            'attention_weights': attn_weights,
            'context_vector': context,
            'logits': logits,
            'probabilities': torch.softmax(logits, dim=1)
        }


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def word_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization"""
    return text.lower().split()

def extract_spans(tokens: List[str], labels: List[str], span_type: str) -> List[str]:
    """Extract BIO spans from tokens and labels"""
    spans = []
    i = 0
    while i < len(labels):
        if labels[i] == f"B-{span_type}":
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{span_type}":
                j += 1
            spans.append(" ".join(tokens[i:j]))
            i = j
        else:
            i += 1
    return spans

def print_tensor_info(name, tensor, show_values=True, max_show=10):
    """Print tensor information"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"   Shape: {tensor.shape}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Device: {tensor.device}")
    if show_values:
        if tensor.numel() <= max_show * 5:
            print(f"   Values:\n{tensor}")
        else:
            print(f"   Values (first {max_show}):\n{tensor.flatten()[:max_show]}...")


# ============================================================================
# 3. LOAD MODELS AND VOCAB
# ============================================================================

print("="*80)
print(" LOADING MODELS AND VOCABULARY")
print("="*80)

# Load vocabulary
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

print(f" Vocabulary size: {len(vocab)}")
print(f"   Sample tokens: {list(vocab.keys())[:10]}")

# Label mappings
label2id = {"O": 0, "B-ASP": 1, "I-ASP": 2, "B-OPI": 3, "I-OPI": 4}
id2label = {i: l for l, i in label2id.items()}
sentiment_map = {0: "Positive", 1: "Neutral", 2: "Negative"}

print(f" Labels: {label2id}")
print(f" Sentiments: {sentiment_map}")

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Device: {device}")

model_crf = BiLSTM_CRF(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=128,
    num_labels=5
).to(device)

model_sent = BiLSTM_Attention(
    vocab_size=len(vocab),
    embed_dim=100,
    hidden_dim=128,
    num_classes=3
).to(device)

# Load weights
model_crf.load_state_dict(torch.load('model_crf.pth', map_location=device))
model_sent.load_state_dict(torch.load('model_sent.pth', map_location=device))
model_crf.eval()
model_sent.eval()

print(" Models loaded successfully!")

# ============================================================================
# 4. PIPELINE DEBUG - STEP BY STEP
# ============================================================================

sample_text = "Absolute waste of time, just adds 4 hours to an already rigorous schedule. Don't waste effort going to any of the classes unless there's a quiz. Assignments are ridiculously easy( git committing questid into a text file)."

print("\n" + "="*80)
print(" STEP 1: SAMPLE TEXT")
print("="*80)
print(f"   Input: '{sample_text}'")

# ----------------------------------------------------------------------------
# STEP 2: Word Tokenization
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print(" STEP 2: WORD TOKENIZATION")
print("="*80)

tokens = word_tokenize(sample_text)
print(f"   Tokens: {tokens}")
print(f"   Length: {len(tokens)} tokens")

# ----------------------------------------------------------------------------
# STEP 3: Map to Vocabulary IDs
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print(" STEP 3: MAP TO VOCABULARY IDs")
print("="*80)

token_ids = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens]
print(f"   Token -> ID mapping:")
for t, id in zip(tokens, token_ids):
    print(f"      '{t}' -> {id}")

input_tensor = torch.tensor([token_ids], device=device)
print_tensor_info("Input Tensor (input_ids)", input_tensor)

# ----------------------------------------------------------------------------
# STEP 4: BiLSTM-CRF Forward Pass
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print(" STEP 4: BiLSTM-CRF FORWARD PASS")
print("="*80)

with torch.no_grad():
    crf_outputs = model_crf.forward_debug(input_tensor)

print("\n 4.1 Embedding Layer")
print_tensor_info("Embedding Output", crf_outputs['embedding'])
print(f"   -> Each token mapped to {crf_outputs['embedding'].shape[2]}-dim vector")

print("\n 4.2 BiLSTM Layer")
print_tensor_info("LSTM Output", crf_outputs['lstm_output'])
print(f"   -> BiLSTM output: 2 directions x {128} hidden = {crf_outputs['lstm_output'].shape[2]}-dim")

print("\n 4.3 Linear (FC) Layer -> Emissions")
print_tensor_info("Emissions (logits for CRF)", crf_outputs['emissions'])
print(f"   -> Each token has {crf_outputs['emissions'].shape[2]} scores (one per label)")

print("\n 4.4 CRF Decode -> Label IDs")
decoded = crf_outputs['decoded'][0]
print(f"   Decoded label IDs: {decoded}")
pred_labels = [id2label[i] for i in decoded]
print(f"   Decoded labels: {pred_labels}")

# Show token-label alignment
print("\n   Token -> Predicted Label:")
for t, l in zip(tokens, pred_labels):
    marker = "[ASP]" if "ASP" in l else ("[OPI]" if "OPI" in l else "[O]")
    print(f"      {marker} '{t}' -> {l}")

# ----------------------------------------------------------------------------
# STEP 5: Extract Spans (Aspects & Opinions)
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print(" STEP 5: EXTRACT SPANS (ASPECTS & OPINIONS)")
print("="*80)

aspects = extract_spans(tokens, pred_labels, "ASP")
opinions = extract_spans(tokens, pred_labels, "OPI")

print(f"   Extracted Aspects: {aspects}")
print(f"   Extracted Opinions: {opinions}")

# ----------------------------------------------------------------------------
# STEP 6: BiLSTM-Attention for Sentiment
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print(" STEP 6: BiLSTM-ATTENTION SENTIMENT PREDICTION")
print("="*80)

with torch.no_grad():
    sent_outputs = model_sent.forward_debug(input_tensor)

print("\n 6.1 Embedding Layer")
print_tensor_info("Embedding Output", sent_outputs['embedding'])

print("\n 6.2 BiLSTM Layer")
print_tensor_info("LSTM Hidden States", sent_outputs['lstm_output'])

print("\n 6.3 Attention Scores (before softmax)")
print_tensor_info("Attention Scores", sent_outputs['attention_scores'])

print("\n 6.4 Attention Weights (after softmax)")
print_tensor_info("Attention Weights", sent_outputs['attention_weights'])
print("\n   Token Attention Weights:")
for t, w in zip(tokens, sent_outputs['attention_weights'][0].cpu().numpy()):
    bar = "#" * int(w * 50)
    print(f"      '{t:<12}' -> {w:.4f} {bar}")

print("\n 6.5 Context Vector (weighted sum)")
print_tensor_info("Context Vector", sent_outputs['context_vector'])
print(f"   -> Represents the entire sentence in {sent_outputs['context_vector'].shape[1]}-dim")

print("\n 6.6 Logits and Probabilities")
print_tensor_info("Logits", sent_outputs['logits'])
print_tensor_info("Probabilities", sent_outputs['probabilities'])

# Get prediction
pred_sent_id = sent_outputs['logits'].argmax(dim=1).item()
pred_sentiment = sentiment_map[pred_sent_id]
probs = sent_outputs['probabilities'][0].cpu().numpy()

print(f"\n   Sentiment Probabilities:")
for i, (sent, prob) in enumerate(zip(['Positive', 'Neutral', 'Negative'], probs)):
    bar = "#" * int(prob * 40)
    marker = "<-- PREDICTED" if i == pred_sent_id else ""
    print(f"      {sent:<10}: {prob:.4f} {bar} {marker}")

print(f"\n   Predicted Sentiment: {pred_sentiment}")

# ----------------------------------------------------------------------------
# STEP 7: Pairing Strategy & Final Triplets
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print(" STEP 7: PAIRING STRATEGY -> FINAL TRIPLETS")
print("="*80)

print("\n   Current pairing: ALL aspects x ALL opinions (Cartesian product)")
print(f"   Aspects: {aspects}")
print(f"   Opinions: {opinions}")
print(f"   Sentiment: {pred_sentiment}")

triplets = []
for asp in aspects:
    for opn in opinions:
        triplet = {
            "aspect": asp,
            "opinion": opn,
            "sentiment": pred_sentiment
        }
        triplets.append(triplet)

print(f"\n   Final Triplets ({len(triplets)} total):")
for i, t in enumerate(triplets, 1):
    print(f"      {i}. Aspect: '{t['aspect']}'")
    print(f"         Opinion: '{t['opinion']}'")
    print(f"         Sentiment: {t['sentiment']}")
    print()

print("\n" + "="*80)
print(" PIPELINE DEBUG COMPLETE!")
print("="*80)   
