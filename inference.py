# ============================================================
# inference.py
# Yêu cầu: đặt file này cùng thư mục với:
#   - transformer.pt
#   - rf_classifier.pkl
#
# Install dependencies:
#   pip install torch scikit-learn
# ============================================================

import pickle
import torch
import torch.nn as nn

# ── Config ────────────────────────────────────────────────
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
POS_EMB_MAX = 256

STRENGTH_LABELS = {
    0: "Rất yếu",
    1: "Yếu",
    2: "Trung bình",
    3: "Khá mạnh",
    4: "Rất mạnh",
}

# ── Load models ───────────────────────────────────────────
checkpoint = torch.load('transformer.pt', map_location=DEVICE)
char2idx   = checkpoint['char2idx']
idx2char   = checkpoint['idx2char']
vocab_size = checkpoint['vocab_size']

with open('rf_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

# ── Model class (phải giống hệt lúc train) ───────────────
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, pos_emb_max=POS_EMB_MAX):
        super().__init__()
        d_model            = 128
        self.pos_emb_max   = pos_emb_max
        self.embedding     = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, pos_emb_max, d_model))
        encoder_layer      = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer   = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc            = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pos_emb_max:
            x       = x[:, :self.pos_emb_max]
            seq_len = self.pos_emb_max
        x_emb = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        mask  = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x_emb.device)
        return self.fc(self.transformer(x_emb, mask=mask, is_causal=True))

model = MiniTransformer(vocab_size).to(DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# ── Internal helpers ──────────────────────────────────────
def _extract_flags(pw: str) -> dict:
    return {
        "LEN":   len(pw),
        "LOWER": int(any(c.islower() for c in pw)),
        "UPPER": int(any(c.isupper() for c in pw)),
        "NUM":   int(any(c.isdigit() for c in pw)),
        "SPEC":  int(any(c in "!@#$%^&*" for c in pw)),
    }

def _generate(start_text: str, max_gen: int = 30, temperature: float = 0.8) -> str:
    input_ids = [char2idx.get(c, 0) for c in start_text]
    for _ in range(max_gen):
        truncated = input_ids[-(POS_EMB_MAX - 1):]
        x_input   = torch.tensor([truncated]).to(DEVICE)
        with torch.no_grad():
            out = model(x_input)
        probs   = torch.softmax(out[0, -1] / temperature, dim=0).cpu()
        next_id = torch.multinomial(probs, 1).item()
        if next_id == 0:
            break
        input_ids.append(next_id)
    result = ''.join([idx2char.get(i, '') for i in input_ids])
    return result.split(":")[-1].strip() if ":" in result else result

# ── Public API ────────────────────────────────────────────
def check_strength(password: str) -> dict:
    """
    Chấm điểm độ mạnh của mật khẩu.

    Input:
        password (str)

    Output:
        {
            "password":       str,
            "strength_score": int,   # 0–4
            "strength_label": str    # "Rất yếu" → "Rất mạnh"
        }
    """
    f     = _extract_flags(password)
    score = int(clf.predict([[f['LEN'], f['LOWER'], f['UPPER'], f['NUM'], f['SPEC']]])[0])
    return {
        "password":       password,
        "strength_score": score,
        "strength_label": STRENGTH_LABELS.get(score, "Không xác định"),
    }


def upgrade_password(password: str, target_len: int) -> dict:
    """
    Nâng cấp mật khẩu yếu thành mật khẩu mạnh hơn.

    Input:
        password   (str) — mật khẩu gốc
        target_len (int) — độ dài mong muốn

    Output:
        {
            "original":       str,
            "enhanced":       str,
            "strength_score": int,   # 0–4
            "strength_label": str
        }
    """
    if target_len <= len(password):
        password = password[:max(1, target_len - 3)]
    chars_to_add = target_len - len(password)

    prompt   = f"<LEN={target_len}><LOWER=1><UPPER=1><NUM=1><SPEC=1>:{password}"
    best_pw, best_score = password, 0

    for _ in range(100):
        enhanced = _generate(prompt, max_gen=chars_to_add)
        if not enhanced or enhanced == password:
            continue
        f     = _extract_flags(enhanced)
        score = int(clf.predict([[f['LEN'], f['LOWER'], f['UPPER'], f['NUM'], f['SPEC']]])[0])
        if score >= 2:
            return {
                "original":       password,
                "enhanced":       enhanced,
                "strength_score": score,
                "strength_label": STRENGTH_LABELS.get(score),
            }
        if score > best_score:
            best_score, best_pw = score, enhanced

    return {
        "original":       password,
        "enhanced":       best_pw,
        "strength_score": best_score,
        "strength_label": STRENGTH_LABELS.get(best_score),
    }
