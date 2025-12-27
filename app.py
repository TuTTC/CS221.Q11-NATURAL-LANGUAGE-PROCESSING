import gradio as gr
import torch
import torch.nn as nn
import pickle
import json
from torchcrf import CRF

# ==========================================
# 1. ĐỊNH NGHĨA MODEL CLASSES (BẮT BUỘC)
# ==========================================
# Phải khớp hoàn toàn với lúc train để load được trọng số
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.pad_idx = pad_idx

    def forward(self, x, tags=None, mask=None):
        if mask is None:
            mask = x != self.pad_idx
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        emissions = self.fc(lstm_out)
        if tags is not None:
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        else:
            return self.crf.decode(emissions, mask=mask)

class BiLSTM_Attention(nn.Module):
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

# ==========================================
# 2. HÀM LOAD RESOURCES
# ==========================================
print("Đang tải resources...")
device = torch.device("cpu") # Deploy chạy CPU

# Load Vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Load Config
with open("config.json", "r") as f:
    config = json.load(f)

# Init Models
model_crf = BiLSTM_CRF(len(vocab), config["embed_dim"], config["hidden_dim"], config["num_labels_crf"])
model_sent = BiLSTM_Attention(len(vocab), config["embed_dim"], config["sent_hidden_dim"], config["num_classes_sent"])

# Load Weights
model_crf.load_state_dict(torch.load("model_crf.pth", map_location=device))
model_sent.load_state_dict(torch.load("model_sent.pth", map_location=device))

model_crf.eval()
model_sent.eval()
print("Resources loaded success!")

# Mappings
LABELS = ["O", "B-ASP", "I-ASP", "B-OPI", "I-OPI"]
id2label = {i: l for i, l in enumerate(LABELS)}
SENTIMENT_MAP = {0: "Positive", 1: "Neutral", 2: "Negative"}

# ==========================================
# 3. LOGIC DỰ ĐOÁN (PIPELINE)
# ==========================================
def predict_fn(text):
    if not text.strip():
        return [], "Vui lòng nhập văn bản."

    # 1. Tiền xử lý
    tokens = text.lower().split()
    input_ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 2. Trích xuất Aspect (CRF)
    mask = (input_tensor != 0)
    with torch.no_grad():
        crf_out = model_crf(input_tensor, mask=mask)
    
    pred_tags = [id2label[tag] for tag in crf_out[0]]

    # 3. Phân tích Cảm xúc (Classification)
    with torch.no_grad():
        logits = model_sent(input_tensor)
        probs = torch.softmax(logits, dim=1) # Lấy xác suất
        top_prob, top_idx = torch.max(probs, dim=1)
        sentiment_label = SENTIMENT_MAP.get(top_idx.item(), "Unknown")
        confidence = top_prob.item()

    # 4. Format Output cho Gradio HighlightedText
    # Gradio yêu cầu list các tuple: [(token, label), (token, label), ...]
    # Nếu label là None thì không tô màu
    highlighted_output = []
    found_aspects = []
    
    current_aspect = []
    
    for token, tag in zip(tokens, pred_tags):
        if tag == "O":
            highlighted_output.append((token, None))
            # Nếu kết thúc aspect thì lưu lại
            if current_aspect:
                found_aspects.append(" ".join(current_aspect))
                current_aspect = []
        else:
            # Làm đẹp nhãn hiển thị (ví dụ: B-ASP -> Aspect)
            display_tag = "Aspect" if "ASP" in tag else "Opinion" if "OPI" in tag else tag
            highlighted_output.append((token, display_tag))
            
            if "ASP" in tag:
                current_aspect.append(token)
    
    # Xử lý aspect cuối câu nếu có
    if current_aspect: found_aspects.append(" ".join(current_aspect))

    # Tạo text kết quả Sentiment
    summary_text = f"**Dự đoán Cảm xúc:** {sentiment_label} ({confidence:.1%})\n\n"
    if found_aspects:
        summary_text += f"**Các khía cạnh (Aspects) tìm thấy:** {', '.join(found_aspects)}"
    else:
        summary_text += "Không tìm thấy Aspect nào."

    return highlighted_output, summary_text

# ==========================================
# 4. GIAO DIỆN GRADIO
# ==========================================
# Tùy chỉnh màu sắc cho tags
color_map = {"Aspect": "#ffae00", "Opinion": "#00bfff"}

with gr.Blocks(title="EduRABSA Demo") as demo:
    gr.Markdown("# 🎓 EduRABSA Analysis Demo")
    gr.Markdown("Hệ thống trích xuất khía cạnh (Aspect) và phân tích cảm xúc sử dụng BiLSTM-CRF & Attention.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Nhập câu review (Tiếng Anh)", 
                placeholder="The teachers are great but the food is bad...",
                lines=3
            )
            btn = gr.Button("Phân tích", variant="primary")
            
            # Examples để người dùng click nhanh
            gr.Examples(
                examples=[
                    ["The professors are knowledgeable but the canteen food is terrible"],
                    ["The library is quiet and conducive to learning"],
                    ["Tuition fees are too high for the poor facilities"]
                ],
                inputs=input_text
            )

        with gr.Column():
            # Output 1: Highlighted Text (Sequence Labeling)
            output_highlight = gr.HighlightedText(
                label="Kết quả Trích xuất",
                combine_adjacent=True,
                show_legend=True,
                color_map=color_map
            )
            
            # Output 2: Sentiment Summary
            output_summary = gr.Markdown(label="Tổng hợp")

    # Sự kiện click
    btn.click(
        fn=predict_fn, 
        inputs=input_text, 
        outputs=[output_highlight, output_summary]
    )

# Launch
if __name__ == "__main__":
    demo.launch(share=True)