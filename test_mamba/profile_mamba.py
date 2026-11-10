import os
import math
import urllib.request
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

DATA_PATH = "tinyshakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def load_data(block_size):
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]

    def get_batch(batch_size, device):
        ix = torch.randint(len(train_data) - block_size - 1, (batch_size,))
        x = torch.stack([train_data[i:i+block_size] for i in ix]).to(device)
        y = torch.stack([train_data[i+1:i+block_size+1] for i in ix]).to(device)
        return x, y
    return get_batch, vocab_size


def main():
    device = "cuda:0"
    batch_size, block_size = 8, 256

    get_batch, vocab_size = load_data(block_size)

    config = MambaConfig(
        d_model=256, n_layer=4, vocab_size=vocab_size,
        ssm_cfg=dict(d_state=16, d_conv=4, expand=2),
        rms_norm=True, fused_add_norm=False,
    )
    model = MambaLMHeadModel(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    # --- Warmup: 5 steps to let CUDA context, cuBLAS handles, etc. stabilize ---
    print("Warming up (5 steps)...")
    for _ in range(5):
        x, y = get_batch(batch_size, device)
        opt.zero_grad(set_to_none=True)
        logits = model(x).logits[..., :vocab_size]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    # --- Profile exactly one step ---
    print("Profiling 1 step...")
    x, y = get_batch(batch_size, device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        opt.zero_grad(set_to_none=True)
        logits = model(x).logits[..., :vocab_size]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    # --- Print sorted by CUDA time ---
    print("\n" + "="*100)
    print("TOP 15 KERNELS BY CUDA TIME")
    print("="*100)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=40,
    ))

    # --- Save a Chrome trace for visual inspection ---
    trace_path = "mamba_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nFull trace saved to: {trace_path}")
    print("Open it at chrome://tracing (or https://ui.perfetto.dev) to inspect the timeline.")


if __name__ == "__main__":
    main()