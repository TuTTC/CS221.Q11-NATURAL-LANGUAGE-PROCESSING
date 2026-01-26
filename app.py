import gradio as gr
import torch
import torch.nn as nn
import pickle
import re
import numpy as np
from torchcrf import CRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification

# ==========================================
# 0. CONFIG & UTILS
# ==========================================
print("Đang khởi tạo thiết bị...")
device = torch.device("cpu") # Chạy CPU cho demo

class TextCleaner:
    def clean_text(self, text):
        if not text: return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

cleaner = TextCleaner()

# ==========================================
# 1. BiLSTM-CRF RESOURCES & LOGIC
# ==========================================
# --- Class Definitions ---
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.pad_idx = pad_idx

    def forward(self, x, tags=None, mask=None):
        if mask is None: mask = x != self.pad_idx
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

# --- Load BiLSTM ---
def load_bilstm():
    print("Loading BiLSTM resources...")
    try:
        with open("models/vocab.pkl", "rb") as f: vocab = pickle.load(f)
        config = {"embed_dim": 100, "hidden_dim": 256, "sent_hidden_dim": 128, "num_labels_crf": 5, "num_classes_sent": 3}
        m_crf = BiLSTM_CRF(len(vocab), config["embed_dim"], config["hidden_dim"], config["num_labels_crf"])
        m_sent = BiLSTM_Attention(len(vocab), config["embed_dim"], config["sent_hidden_dim"], config["num_classes_sent"])
        m_crf.load_state_dict(torch.load("models/BiLSTM/model_crf.pth", map_location=device))
        m_sent.load_state_dict(torch.load("models/BiLSTM/model_sent.pth", map_location=device))
        m_crf.eval(); m_sent.eval()
        return m_crf, m_sent, vocab
    except Exception as e:
        print(f"BiLSTM Load Error: {e}"); return None, None, None

# --- Logic BiLSTM ---
LABELS_BILSTM = ["O", "B-ASP", "I-ASP", "B-OPI", "I-OPI"]
id2label_bilstm = {i: l for i, l in enumerate(LABELS_BILSTM)}
SENTIMENT_MAP_BILSTM = {0: "Positive", 1: "Neutral", 2: "Negative"}

def predict_bilstm(text):
    if bilstm_crf is None: return [], "Model Error", []
    
    # 1. Inference (Vẫn dùng lowercase để model hiểu)
    tokens_lower = text.lower().split()
    input_ids = [vocab.get(t, vocab.get("<UNK>", 0)) for t in tokens_lower]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        crf_out = bilstm_crf(input_tensor, mask=input_tensor!=0)
        logits = bilstm_sent(input_tensor)
    
    pred_tags = [id2label_bilstm[tag] for tag in crf_out[0]]
    sentiment_label = SENTIMENT_MAP_BILSTM.get(torch.max(torch.softmax(logits, dim=1), dim=1)[1].item(), "Unknown")
    
    # 2. Visualize Logic (SỬA ĐỔI: Dùng token gốc để hiển thị)
    original_tokens = text.split() # Token giữ nguyên hoa thường/dấu câu
    
    highlighted = []
    aspects, opinions = [], []
    curr_phrase, curr_type = [], None
    
    # Zip token gốc với nhãn dự đoán
    # Lưu ý: Giả định tokenizer của BiLSTM là split() cơ bản nên độ dài khớp nhau
    for token, tag in zip(original_tokens, pred_tags):
        tag_type = "ASP" if "ASP" in tag else "OPI" if "OPI" in tag else None
        
        # Highlight Logic
        if tag == "O":
            highlighted.append((token, None))
            if curr_phrase: 
                # Lưu phrase cũ
                phrase_str = " ".join(curr_phrase)
                if curr_type == "ASP": aspects.append(phrase_str)
                elif curr_type == "OPI": opinions.append(phrase_str)
                curr_phrase = []
        else:
            # Tô màu (Aspect/Opinion)
            display = "Aspect" if "ASP" in tag else "Opinion"
            highlighted.append((token, display))
            
            # Logic trích xuất phrase (B- / I-)
            if tag.startswith("B-"):
                if curr_phrase: # Đóng phrase trước đó nếu có
                    phrase_str = " ".join(curr_phrase)
                    if curr_type == "ASP": aspects.append(phrase_str)
                    elif curr_type == "OPI": opinions.append(phrase_str)
                curr_phrase = [token]; curr_type = tag_type
            elif tag.startswith("I-") and curr_type == tag_type:
                curr_phrase.append(token)
    
    # Xử lý phrase cuối cùng
    if curr_phrase:
        phrase_str = " ".join(curr_phrase)
        if curr_type == "ASP": aspects.append(phrase_str)
        elif curr_type == "OPI": opinions.append(phrase_str)

    # 3. Table Logic (Pairing)
    table = []
    max_len = max(len(aspects), len(opinions))
    for i in range(max_len):
        a = aspects[i] if i < len(aspects) else "NULL"
        o = opinions[i] if i < len(opinions) else "NULL"
        # BiLSTM dùng sentiment chung cho cả câu
        table.append([a, o, sentiment_label])

    return highlighted, f"**Sentiment:** {sentiment_label}", table

# ==========================================
# 2. T5 RESOURCES & LOGIC (ĐÃ SỬA LỖI VISUALIZE)
# ==========================================
def load_t5():
    print("Loading T5 resources...")
    try:
        path = "models/t5_processed_aste_best"
        tok = AutoTokenizer.from_pretrained(path)
        mod = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
        return tok, mod
    except Exception as e: print(f"T5 Load Error: {e}"); return None, None

def predict_t5(text):
    if t5_model is None: return [], "Model Error", []
    
    # 1. Generate Text
    clean_text = cleaner.clean_text(text)
    inputs = t5_tokenizer("extract triplets: " + clean_text, return_tensors="pt", max_length=256, truncation=True).to(device)
    with torch.no_grad(): 
        out = t5_model.generate(**inputs, max_length=256)
    gen_text = t5_tokenizer.decode(out[0], skip_special_tokens=True)
    
    # 2. Parse Triplets
    triplets = []
    if gen_text:
        # Tách các bộ ba ngăn cách bởi dấu chấm phẩy
        for item in gen_text.split(';'):
            # Xử lý các dấu ngoặc thừa
            item = item.strip()
            if item.startswith('(') and item.endswith(')'):
                item = item[1:-1]
            
            parts = item.split(',')
            if len(parts) == 3: 
                triplets.append([p.strip() for p in parts])
    
    # 3. Map Highlight (Logic Mới: Chính xác hơn)
    # Tách token từ câu gốc (giữ nguyên dấu câu để hiển thị)
    original_tokens = text.split()
    
    # Tạo danh sách token sạch (chữ thường, bỏ dấu câu) để so sánh
    clean_tokens_map = []
    for t in original_tokens:
        # Xóa hết ký tự đặc biệt, chỉ giữ chữ và số
        t_clean = re.sub(r'[^\w\s]', '', t).lower()
        clean_tokens_map.append(t_clean)
        
    mask = [None] * len(original_tokens)
    
    # Gom tất cả từ vựng trong Aspect và Opinion mà T5 tìm được
    aspect_words = set()
    opinion_words = set()
    
    for asp, opi, _ in triplets:
        if asp != 'null':
            aspect_words.update(asp.lower().split())
        if opi != 'null':
            opinion_words.update(opi.lower().split())
            
    # So sánh khớp từ (Exact Match)
    for i, t_clean in enumerate(clean_tokens_map):
        if not t_clean: continue # Bỏ qua nếu token chỉ toàn dấu câu
        
        # Ưu tiên Aspect trước
        if t_clean in aspect_words:
            mask[i] = "Aspect"
        # Sau đó đến Opinion (nếu từ đó vừa là asp vừa là opi, code này ưu tiên asp)
        elif t_clean in opinion_words:
            mask[i] = "Opinion"
    
    # Ghép token gốc với nhãn màu
    highlighted_output = [(t, l) for t, l in zip(original_tokens, mask)]
    
    return highlighted_output, gen_text, triplets

# ==========================================
# 3. BERT RESOURCES & LOGIC (OPTIMIZED)
# ==========================================

def load_bert():
    print("Loading BERT resources...")
    try:
        model_path = "models/bert_aste_final"
        
        # Load Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        
        # Lấy id2label chuẩn từ config của model (QUAN TRỌNG)
        # Giúp tránh sai lệch thứ tự nhãn
        id2label = model.config.id2label 
        
        return tokenizer, model, id2label
    except Exception as e:
        print(f"Error loading BERT: {e}")
        return None, None, {}

def extract_entities(tokens, preds, id2label):
    """
    Hàm hỗ trợ trích xuất entities theo chuẩn BIO từ token và prediction
    """
    entities = [] # List of dict: {'text': string, 'type': ASP/OPI, 'sentiment': POS/NEG/NEU, 'start': idx, 'end': idx}
    curr_entity = None
    
    for i, (token, label_idx) in enumerate(zip(tokens, preds)):
        if token in ["[CLS]", "[SEP]", "[PAD]"]: continue
        
        label = id2label[label_idx] # Lấy nhãn từ config
        
        # Bỏ qua subword '##' khi xét nhãn bắt đầu, nhưng vẫn dùng để ghép text
        is_subword = token.startswith("##")
        clean_token = token.replace("##", "")
        
        if label.startswith("B-"):
            # Lưu entity cũ nếu có
            if curr_entity: entities.append(curr_entity)
            
            # Tạo entity mới
            e_type = "ASP" if "ASP" in label else "OPI"
            sent = label.split("-")[-1] if e_type == "OPI" else None
            curr_entity = {
                "tokens": [clean_token],
                "type": e_type,
                "sentiment": sent,
                "start": i, # Lưu vị trí để tính khoảng cách
                "end": i
            }
            
        elif label.startswith("I-") and curr_entity:
            # Kiểm tra xem có khớp loại không (ASP đi với I-ASP)
            match_type = ("ASP" in label and curr_entity['type'] == "ASP") or \
                         ("OPI" in label and curr_entity['type'] == "OPI")
            
            if match_type:
                curr_entity["tokens"].append(clean_token)
                curr_entity["end"] = i
            else:
                # Nếu đang là I- mà không khớp (vd đang ASP mà gặp I-OPI), ngắt luôn
                entities.append(curr_entity)
                curr_entity = None
        
        else: # Label O
            if curr_entity:
                entities.append(curr_entity)
                curr_entity = None
                
    if curr_entity: entities.append(curr_entity)
    
    # Join tokens lại thành từ hoàn chỉnh
    for e in entities:
        e["text"] = " ".join(e["tokens"]).replace(" .", ".").replace(" ,", ",") # Simple fix punctuation
        
    return entities

def predict_bert(text):
    # Đảm bảo các biến global đã được load
    if bert_model is None: return [], "Model Error", []
    
    # 1. Inference
    # Lưu ý: id2label được load từ lúc gọi load_bert() hoặc truy cập từ model.config.id2label
    id2label = bert_model.config.id2label
    
    clean_text = cleaner.clean_text(text)
    inputs = bert_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    tokens = bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # 2. Extract Entities (BIO Logic)
    entities = extract_entities(tokens, predictions, id2label)
    
    aspects = [e for e in entities if e['type'] == 'ASP']
    opinions = [e for e in entities if e['type'] == 'OPI']
    
    # 3. Tạo Highlight cho UI
    # Mapping lại subwords thành words để hiển thị (Logic đơn giản hóa)
    # Ta sẽ duyệt qua tokens và gom nhóm dựa trên entities đã tìm được
    
    final_highlight = []
    
    # Để hiển thị đẹp, ta cần map lại toàn bộ text
    # Cách đơn giản: Reconstruct từ tokens và gán nhãn
    curr_idx = 0
    while curr_idx < len(tokens):
        token = tokens[curr_idx]
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            curr_idx += 1
            continue
            
        # Kiểm tra token này có thuộc entity nào không
        found_entity = None
        for e in entities:
            if e['start'] <= curr_idx <= e['end']:
                found_entity = e
                break
        
        if found_entity:
            # Nếu thuộc entity, thêm cả cụm entity vào highlight
            label_display = "Aspect" if found_entity['type'] == "ASP" else "Opinion"
            final_highlight.append((found_entity['text'], label_display))
            # Nhảy index qua hết entity này
            curr_idx = found_entity['end'] + 1
        else:
            # Nếu là từ thường (O), thêm vào text thường
            word = token.replace("##", "")
            # Nếu phần tử trước đó là string thường, nối vào để đỡ bị nát
            if final_highlight and isinstance(final_highlight[-1], str):
                # Xử lý dấu cách logic subword
                if token.startswith("##"):
                    final_highlight[-1] += word
                else:
                    final_highlight[-1] += " " + word
            else:
                final_highlight.append(word)
            curr_idx += 1

    # 4. Triplet Pairing (Nearest Neighbor Heuristic)
    # Logic: Với mỗi Aspect, tìm Opinion có khoảng cách index gần nhất
    triplets = []
    sent_map = {"POS": "Positive", "NEG": "Negative", "NEU": "Neutral", "Unknown": "Unknown"}

    if not aspects and not opinions:
        pass # Không có gì
    elif aspects and not opinions:
        for a in aspects: triplets.append([a['text'], "NULL", "Unknown"])
    elif not aspects and opinions:
        for o in opinions: triplets.append(["NULL", o['text'], sent_map.get(o['sentiment'], "Unknown")])
    else:
        # Có cả 2 -> Ghép đôi dựa trên khoảng cách
        used_opinions = set()
        
        for asp in aspects:
            # Tìm opinion gần nhất (tính bằng khoảng cách giữa index trung tâm)
            asp_center = (asp['start'] + asp['end']) / 2
            
            best_opi = None
            min_dist = 9999
            
            for opi in opinions:
                opi_center = (opi['start'] + opi['end']) / 2
                dist = abs(asp_center - opi_center)
                if dist < min_dist:
                    min_dist = dist
                    best_opi = opi
            
            if best_opi:
                triplets.append([
                    asp['text'], 
                    best_opi['text'], 
                    sent_map.get(best_opi['sentiment'], "Unknown")
                ])
                # Note: Một opinion có thể bổ nghĩa cho nhiều aspect nên ta KHÔNG remove best_opi khỏi danh sách

    summary = f"**BERT Found:** {len(aspects)} Aspects, {len(opinions)} Opinions"
    return final_highlight, summary, triplets


# ==========================================
# 4. INIT MODELS
# ==========================================
bilstm_crf, bilstm_sent, vocab = load_bilstm()
t5_tokenizer, t5_model = load_t5()
bert_tokenizer, bert_model = load_bert()

# ==========================================
# 5. GRADIO UI
# ==========================================
def analyze_all(text):
    bi_hl, bi_sum, bi_tab = predict_bilstm(text)
    t5_hl, t5_raw, t5_tab = predict_t5(text)
    bert_hl, bert_sum, bert_tab = predict_bert(text)
    return bi_hl, bi_sum, bi_tab, t5_hl, t5_raw, t5_tab, bert_hl, bert_sum, bert_tab

# CSS cố định chiều cao bảng (được load thông qua Blocks)
css = ".dataframe-wrap { max_height: 200px; overflow-y: auto; }"

# XÓA layout="wide" ĐỂ FIX LỖI
with gr.Blocks(title="NLP Triple Benchmark (Task ASTE)", css=css) as demo:
    gr.Markdown("# ⚔️ NLP Model Arena: BiLSTM vs. T5 vs. BERT")
    gr.Markdown("So sánh 3 kiến trúc mô hình khác nhau trên cùng bài toán ASTE.")
    
    with gr.Row():
        input_text = gr.Textbox(label="Input Review", placeholder="The staff is friendly but the prices are high.", lines=2)
        btn = gr.Button("🚀 RUN", variant="primary")

    with gr.Row():
        # --- BiLSTM ---
        with gr.Column(variant="panel"):
            gr.Markdown("### 1. BiLSTM-CRF (Classic)")
            bi_hl = gr.HighlightedText(label="Tags", color_map={"Aspect": "#ffae00", "Opinion": "#00bfff"})
            bi_tab = gr.Dataframe(headers=["Aspect", "Opinion", "Sentiment"], datatype=["str", "str", "str"])
            bi_sum = gr.Markdown()
            
        # --- T5 ---
        with gr.Column(variant="panel"):
            gr.Markdown("### 2. T5-Base (Generative)")
            t5_hl = gr.HighlightedText(label="Mapped Tags", color_map={"Aspect": "#ffae00", "Opinion": "#00bfff"})
            t5_tab = gr.Dataframe(headers=["Aspect", "Opinion", "Sentiment"], datatype=["str", "str", "str"])
            t5_raw = gr.Textbox(label="Raw Output", lines=1)

        # --- BERT ---
        with gr.Column(variant="panel"):
            gr.Markdown("### 3. BERT (Encoder)")
            bert_hl = gr.HighlightedText(label="Tags (Subword Merged)", color_map={"Aspect": "#ffae00", "Opinion": "#00bfff"})
            bert_tab = gr.Dataframe(headers=["Aspect", "Opinion", "Sentiment"], datatype=["str", "str", "str"])
            bert_sum = gr.Markdown()

    gr.Examples([
                    ["The professors are knowledgeable but the canteen food is terrible"],
                    ["The library is quiet and conducive to learning"],
                    ["Tuition fees are too high for the poor facilities"]
                ], inputs=input_text)

    btn.click(analyze_all, inputs=input_text, 
              outputs=[bi_hl, bi_sum, bi_tab, t5_hl, t5_raw, t5_tab, bert_hl, bert_sum, bert_tab])

if __name__ == "__main__":
    demo.launch(share=True)