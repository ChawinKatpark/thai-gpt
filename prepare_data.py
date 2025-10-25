import os
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# Folder containing your .txt summaries
DATA_DIR = "summaries"
SAVE_PATH = "data_thai_gpt.pt"

# 1️⃣ Load Thai tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "airesearch/wangchanberta-base-att-spm-uncased"
)

print("✅ Tokenizer loaded:", tokenizer.name_or_path)

# 2️⃣ Read all text files and combine into one long string
texts = []
for fname in tqdm(os.listdir(DATA_DIR), desc="Reading files"):
    if fname.endswith(".txt"):
        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:  # skip empty
                texts.append(text)

print(f"✅ Loaded {len(texts)} documents")

# 3️⃣ Join with special separator (so sentences don't merge oddly)
full_text = "\n".join(texts)

# 4️⃣ Tokenize entire dataset
tokens = tokenizer.encode(full_text)

print(f"✅ Tokenized length: {len(tokens):,} tokens")

# 5️⃣ Convert to tensor
data = torch.tensor(tokens, dtype=torch.long)

# 6️⃣ Split into train and validation sets (90/10)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.save((train_data, val_data), SAVE_PATH)
print(f"💾 Saved dataset → {SAVE_PATH}")
