import os
import math
import time
import urllib.request
import torch
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.generation import InferenceParams

# -----------------------------
# 1. Dataset (TinyShakespeare)
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "tinyshakespeare.txt"


def load_data(block_size):
    if not os.path.exists(DATA_PATH):
        print(f"Downloading {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split, batch_size, device):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix]).to(device)
        y = torch.stack([d[i + 1:i + block_size + 1] for i in ix]).to(device)
        return x, y

    return get_batch, vocab_size, chars


# -----------------------------
# 2. Training Loop
# -----------------------------
def main():
    device = "cuda:0"
    batch_size = 8
    block_size = 256
    max_steps = 500
    eval_interval = 50

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

    get_batch, vocab_size, chars = load_data(block_size)
    print(f"Vocab size: {vocab_size}")

    # Tiny Mamba-1 LM
    config = MambaConfig(
        d_model=256,
        n_layer=4,
        vocab_size=vocab_size,
        ssm_cfg=dict(d_state=16, d_conv=4, expand=2),
        rms_norm=True,
        fused_add_norm=False,  # explicit: no Triton on Kepler
    )
    model = MambaLMHeadModel(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    pad_vocab = model.lm_head.out_features
    print(f"Model parameters: {n_params/1e6:.2f}M")
    print(f"Padded vocab size (lm_head out): {pad_vocab}")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))

    # Cosine LR schedule with warmup
    def get_lr(step):
        warmup = 50
        if step < warmup:
            return 3e-4 * (step + 1) / warmup
        progress = (step - warmup) / (max_steps - warmup)
        return 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * progress))

    model.train()
    t0 = time.time()

    print("\nStarting training...")
    print("Step | Loss   | LR      | Time")
    print("-" * 40)

    for step in range(max_steps):
        lr = get_lr(step)
        for g in opt.param_groups:
            g['lr'] = lr

        x, y = get_batch("train", batch_size, device)

        opt.zero_grad(set_to_none=True)

        # NOTE: We use FP32 here. Kepler lacks fast FP16 CUDA cores.
        # If we used autocast(FP16), it might actually be slower due to conversions.
        # Slice to real vocab_size to ignore the padding rows in lm_head.
        logits = model(x).logits[..., :vocab_size]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_interval == 0 or step == max_steps - 1:
            dt = time.time() - t0
            t0 = time.time()
            print(f"{step:4d} | {loss.item():.4f} | {lr:.2e} | {dt:.1f}s")

            # Validation
            if step > 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for _ in range(10):
                        vx, vy = get_batch("val", batch_size, device)
                        vlogits = model(vx).logits[..., :vocab_size]
                        val_loss += F.cross_entropy(vlogits.reshape(-1, vocab_size), vy.reshape(-1)).item()
                val_loss /= 10
                print(f"     | VAL: {val_loss:.4f}")
                model.train()

    print("\nTraining complete.")

    # -----------------------------
    # 3. Sample Generation
    # -----------------------------
    print("\n--- Sample Generation ---")
    model.eval()
    prompt = "ROMEO:"
    stoi = {c: i for i, c in enumerate(chars)}
    idx = torch.tensor([[stoi[c] for c in prompt]], device=device)

    with torch.no_grad():
        inf = InferenceParams(max_seqlen=256, max_batch_size=1)
        out = model(idx, inference_params=inf)
        inf.seqlen_offset += idx.shape[1]

        generated = prompt
        for _ in range(100):
            # Slice to real vocab_size so sampling only picks real characters
            logits = out.logits[:, -1, :vocab_size]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated += chars[next_token.item()]

            # Step the model
            out = model(next_token, inference_params=inf)
            inf.seqlen_offset += 1

    print(generated)


if __name__ == "__main__":
    main()