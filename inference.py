import os
import pickle
import random
import torch
import torch.nn as nn

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
POS_EMB_MAX = 256

STRENGTH_LABELS = {
    0: "Rất yếu",
    1: "Yếu",
    2: "Trung bình",
    3: "Khá mạnh",
    4: "Rất mạnh",
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

checkpoint = torch.load(os.path.join(CURRENT_DIR, 'transformer.pt'), map_location=DEVICE)
char2idx   = checkpoint['char2idx']
idx2char   = checkpoint['idx2char']
vocab_size = checkpoint['vocab_size']

with open(os.path.join(CURRENT_DIR, 'rf_classifier.pkl'), 'rb') as f:
    clf = pickle.load(f)


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


FUNNY_ADJECTIVES = [
    "Cosmic", "Neon", "Cyber", "Quantum", "Funky", "Hidden", "Frosty", "Lunar", "Solar", "Epic",
    "Magic", "Crazy", "Sneaky", "Flying", "Super", "Mega", "Hyper", "Secret", "Golden", "Silver",
    "Toxic", "Spicy", "Sweet", "Bitter", "Sour", "Salty", "Wild", "Tame", "Brave", "Shy",
    "Swift", "Slow", "Heavy", "Light", "Dark", "Bright", "Cold", "Hot", "Cool", "Warm",
    "Loud", "Quiet", "Rough", "Smooth", "Hard", "Soft", "Sharp", "Dull", "Big", "Small",
    "Tall", "Short", "Thick", "Flat", "Round", "Square", "Blue", "Red", "Green", "Yellow",
    "Orange", "Purple", "Pink", "Black", "White", "Gray", "Brown", "Cyan", "Teal", "Aqua",
    "Jade", "Ruby", "Opal", "Onyx", "Iron", "Steel", "Brass", "Copper", "Stone", "Wood",
    "Fiery", "Windy", "Earthy", "Starry", "Sunny", "Void", "Spacial", "Mental", "Soulful", "Bony",
    "Savage", "Brutal", "Cunning", "Sly", "Clever", "Smart", "Genius", "Brilliant", "Creepy", "Spooky",
    "Scary", "Ghostly", "Ghastly", "Haunted", "Holy", "Sacred", "Divine", "Godly", "Angelic", "Demonic",
    "Celestial", "Galactic", "Universal", "Alien", "Mutant", "Cyborg", "Robotic", "Atomic", "Radiant", "Glowing",
    "Awesome", "Radical", "Bodacious", "Gnarly", "Wicked", "Righteous", "Stellar", "Mighty", "Fierce", "Playful",
]

FUNNY_NOUNS = [
    "Penguin", "Dragon", "Ninja", "Wizard", "Cactus", "Phoenix", "Rocket", "Robot", "Kraken", "Tiger",
    "Lion", "Bear", "Wolf", "Fox", "Shark", "Whale", "Crab", "Mushroom", "Goblin", "Vampire",
    "Orc", "Troll", "Ogre", "Giant", "Fairy", "Pixie", "Sprite", "Gnome", "Dwarf", "Elf",
    "Demon", "Angel", "Ghost", "Ghoul", "Zombie", "Mummy", "Cyborg", "Mutant", "Monster", "Beast",
    "Hero", "Villain", "King", "Queen", "Knight", "Wizard", "Rogue", "Bard", "Cleric", "Paladin",
    "Sword", "Shield", "Crown", "Gem", "Jewel", "Rocket", "Comet", "Nebula", "Quasar", "Pulsar",
]


def _extract_flags(pw: str) -> dict:
    return {
        "LEN":   len(pw),
        "LOWER": int(any(c.islower() for c in pw)),
        "UPPER": int(any(c.isupper() for c in pw)),
        "NUM":   int(any(c.isdigit() for c in pw)),
        "SPEC":  int(any(c in "!@#$%^&*()_+-=[]{};':\\|,.<>/?" for c in pw)),
    }


def _score(pw: str) -> int:
    f = _extract_flags(pw)
    return int(clf.predict([[f['LEN'], f['LOWER'], f['UPPER'], f['NUM'], f['SPEC']]])[0])


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

    if ":" in result:
        colon_idx = result.index(":")
        return result[colon_idx + 1:].strip()
    return result


def _make_funny_and_leet(word: str, inc_upper: bool, inc_num: bool, inc_sym: bool) -> str:
    if not word or not word.strip():
        word = random.choice(FUNNY_ADJECTIVES) + random.choice(FUNNY_NOUNS)

    if inc_upper:
        word = "".join(random.choice([c.upper(), c.lower()]) for c in word)
    else:
        word = word.lower()

    leet_map = {}
    if inc_sym: leet_map.update({'a': '@', 'i': '!', 's': '$'})
    if inc_num: leet_map.update({'e': '3', 'o': '0', 't': '7', 'b': '8', 'g': '9'})

    return "".join(leet_map.get(c.lower(), c) for c in word)


def _pick_pad_char(inc_upper: bool, inc_num: bool, inc_sym: bool) -> str:
    if inc_sym:   return random.choice(['#', '-', '*', '!', '@'])
    elif inc_num: return str(random.randint(2, 9))
    elif inc_upper: return random.choice(['X', 'Z', 'Q'])
    else:         return 'x'


def _enforce_constraints(pw: str, inc_upper: bool, inc_num: bool,
                          inc_sym: bool, inc_ambig: bool) -> str:
    if not inc_upper: pw = pw.lower()
    if not inc_num:   pw = ''.join(c for c in pw if not c.isdigit())
    if not inc_sym:   pw = ''.join(c for c in pw if c.isalnum())
    if inc_ambig:
        for ch in "il1Lo0O":
            pw = pw.replace(ch, "z")
    return pw


def process_password(password: str, target_len: int = 16,
                     inc_upper: bool = True, inc_num: bool = True,
                     inc_sym: bool = True, inc_ambig: bool = False) -> dict:

    transformed_base = _make_funny_and_leet(password, inc_upper, inc_num, inc_sym)

    if inc_ambig:
        for ch in "il1Lo0O":
            transformed_base = transformed_base.replace(ch, "x")

    if target_len <= len(transformed_base):
        transformed_base = transformed_base[:max(1, target_len - 3)]

    chars_to_add = target_len - len(transformed_base)

    prompt = (
        f"<LEN={target_len}>"
        f"<LOWER=1>"
        f"<UPPER={int(inc_upper)}>"
        f"<NUM={int(inc_num)}>"
        f"<SPEC={int(inc_sym)}>:{transformed_base}"
    )

    best_enhanced = transformed_base
    best_score    = _score(transformed_base)

    max_possible_score = 4
    if not inc_upper: max_possible_score -= 1
    if not inc_num:   max_possible_score -= 1
    if not inc_sym:   max_possible_score -= 1

    for _ in range(50):
        enhanced = _generate(prompt, max_gen=chars_to_add)

        if not enhanced or len(enhanced) <= len(transformed_base):
            continue

        enhanced = _enforce_constraints(enhanced, inc_upper, inc_num, inc_sym, inc_ambig)

        while len(enhanced) < target_len:
            enhanced += _pick_pad_char(inc_upper, inc_num, inc_sym)

        enhanced = enhanced[:target_len]

        score = _score(enhanced)

        if score > best_score:
            best_score    = score
            best_enhanced = enhanced

        if best_score >= max_possible_score:
            break

    return {
        "original":       password,
        "enhanced":       best_enhanced,
        "strength_score": best_score,
        "strength_label": STRENGTH_LABELS.get(best_score, "Không xác định"),
    }
