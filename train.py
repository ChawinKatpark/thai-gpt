import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import os
import json
from datetime import datetime
from google.colab import drive
import sys
import itertools
import re

# Add the directory containing model.py to the system path
sys.path.append('/content/thai-gpt')
from model import GPTLanguageModel, GPTConfig


# Mount Drive
drive.mount('/content/drive', force_remount=True)
save_dir = "/content/drive/MyDrive/thai_gpt/checkpoints"
os.makedirs(save_dir, exist_ok=True)
print(f"💾 Checkpoints will be saved to: {save_dir}")


# Dataset Class
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + 1 + self.block_size]
        return x, y

# Config
block_size = 128
batch_size = 32
max_iters = 100000
eval_interval = 2000
checkpoint_interval = 4000
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"you use device: {device}")

# Load Data
train_data, val_data = torch.load("/content/drive/MyDrive/thai_gpt/data_thai_gpt.pt")
train_dataset = CharDataset(train_data, block_size)
val_dataset = CharDataset(val_data, block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
vocab_size = 30000

# Model + Optimizer
config = GPTConfig(vocab_size, block_size, n_embd=128, n_head=4, n_layer=4, dropout=0.1)
model = GPTLanguageModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Resume if checkpoint exists (fix: numerical sorting)
latest_ckpt = None
step_numbers = []

for file in os.listdir(save_dir):
    match = re.search(r"step(\d+)\.pt", file)
    if match:
        step_numbers.append((int(match.group(1)), file))

if step_numbers:
    step_numbers.sort()
    latest_ckpt = os.path.join(save_dir, step_numbers[-1][1])

start_step = 0
if latest_ckpt:
    print(f"🔄 Loading checkpoint {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"] + 1
    print(f"✅ Resumed from step {start_step}")
else:
    print("🆕 No checkpoint found. Starting fresh training.")

# Evaluation Function (used only at the end)
@torch.no_grad()
def evaluate_val():
    model.eval()
    losses = []
    for xb, yb in tqdm(val_loader, desc="🔍 Eval", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# Train Loop
log_file = os.path.join(save_dir, "training_log.jsonl")
progress = tqdm(range(start_step, max_iters), desc="Training", position=0, leave=True)

# Infinite iterator to avoid StopIteration
train_iter = iter(itertools.cycle(train_loader))

for step in progress:
    xb, yb = next(train_iter)
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    progress.set_postfix({"loss": f"{loss.item():.4f}"})

    # Save checkpoint periodically
    if step > 0 and step % checkpoint_interval == 0:
        ckpt_path = os.path.join(save_dir, f"thai_gpt_model_step{step}.pt")
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)
        # Log progress
        entry = {
            "time": datetime.now().isoformat(),
            "step": step,
            "train_loss": float(loss.item())
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

# Evaluate once at the end (prevent short lifetime GPU)
val_loss = evaluate_val()
entry = {
    "time": datetime.now().isoformat(),
    "step": max_iters,
    "train_loss": float(loss.item()),
    "val_loss": float(val_loss),
}
with open(log_file, "a") as f:
    f.write(json.dumps(entry) + "\n")

print(f"\n🧾 Final step {max_iters}: train {loss.item():.4f}, val {val_loss:.4f}")

final_ckpt_path = os.path.join(save_dir, f"thai_gpt_model_final_step{max_iters}.pt")
torch.save({
    "step": max_iters,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}, final_ckpt_path)
print(f"Saved final checkpoint at {final_ckpt_path}")

print("Training complete!")