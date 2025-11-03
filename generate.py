import torch
from transformers import AutoTokenizer
from model import GPTLanguageModel, GPTConfig
import os
import re

# Load Tokenizer (same as during training)
tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")

# Load Model Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 30000  # same as training config

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=128,
    n_embd=128,
    n_head=4,
    n_layer=4,
    dropout=0.1,
)

# Find the Latest Checkpoint
save_dir = "/content/drive/MyDrive/thai_gpt/checkpoints"

def extract_step(filename):
    match = re.search(r"step(\d+)", filename)
    return int(match.group(1)) if match else -1

ckpts = [f for f in os.listdir(save_dir) if f.endswith(".pt") and "step" in f]
if not ckpts:
    raise FileNotFoundError(f"No checkpoint files found in {save_dir}")

latest_ckpt = max(ckpts, key=extract_step)
ckpt_path = os.path.join(save_dir, latest_ckpt)

# Load Model and Weights
model = GPTLanguageModel(config).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"✅ Loaded model from: {ckpt_path}")

# Custom Text Generation Function
@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.9, top_k=50, top_p=0.9):
    """
    Custom text generation loop for your GPTLanguageModel.
    """
    model.eval()
    device = next(model.parameters()).device

    # Encode the prompt into token IDs
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        # Keep only the last block_size tokens
        input_cond = input_ids[:, -config.block_size:]

        # Forward pass
        logits, _ = model(input_cond)

        # Focus on the last token's logits
        logits = logits[:, -1, :] / temperature

        # Top-k and Top-p sampling (optional)
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float("Inf")

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_logits[sorted_indices_to_remove] = -float("Inf")

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Sample the next token
        next_token = torch.multinomial(probs, num_samples=1)

        # Append next token to the sequence
        input_ids = torch.cat((input_ids, next_token), dim=1)

    # Decode the entire sequence into text
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# Try Generating Text
prompt = ""
generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=100)

print("\n📝 Prompt:", prompt)
print("💬 Generated:", generated_text)