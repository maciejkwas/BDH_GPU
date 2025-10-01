import os
from typing import Dict

import torch

from .model import BDH_GPU
from .config_utils import load_config
from .tokenizer import SimpleTokenizer


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    return ckpt


def sample_next(
    logits: torch.Tensor, temperature: float, top_k: int | None, top_p: float | None
) -> torch.Tensor:
    # logits: (V,)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    # top-k
    if top_k is not None and top_k > 0 and top_k < probs.numel():
        topk_vals, topk_idx = torch.topk(probs, top_k)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[topk_idx] = False
        probs = probs.masked_fill(mask, 0)
        probs = probs / probs.sum()
    # top-p (nucleus)
    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumsum > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0)
        probs = torch.zeros_like(probs)
        probs[sorted_idx] = sorted_probs
        probs = probs / probs.sum()
    idx = torch.multinomial(probs, num_samples=1)
    return idx


def generate(
    ckpt_path: str,
    prompt: str,
    steps: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: str,
) -> str:
    ckpt = load_checkpoint(ckpt_path, device)
    stoi = ckpt.get("stoi")
    itos = ckpt.get("itos")
    tokenizer_json = ckpt.get("tokenizer_json")
    vocab_size = ckpt["vocab_size"]
    cfg = ckpt["config"]

    model = BDH_GPU(
        vocab_size=vocab_size,
        D=cfg["D"],
        H=cfg["H"],
        N=cfg["N"],
        L=cfg["L"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Encode prompt
    use_tokenizer = tokenizer_json is not None
    tk = None
    if use_tokenizer:
        tk = SimpleTokenizer.from_json_str(tokenizer_json)
        tokens = tk.encode_ids(prompt, add_bos_eos=False)
        x = torch.tensor(tokens or [0], dtype=torch.long, device=device).unsqueeze(0)
    else:
        tokens = [stoi.get(ch, 0) for ch in prompt] if stoi else []
        if len(tokens) == 0:
            tokens = [0]
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    with torch.no_grad():
        for _ in range(steps):
            logits = model(x)[0, -1]  # (V,)
            idx = sample_next(logits, temperature, top_k, top_p)
            x = torch.cat([x, idx.view(1, 1)], dim=1)

    ids = x[0].tolist()
    if use_tokenizer and tk is not None:
        out = tk.decode_ids(ids)
    else:
        out = "".join(itos[int(i)] for i in ids)
    return out


def main():
    cfg = load_config()
    gcfg = cfg["generate"]
    top_k = None if int(gcfg.get("top_k", 0)) <= 0 else int(gcfg["top_k"])
    tp = float(gcfg.get("top_p", 0.0))
    top_p = None if tp <= 0 else tp
    txt = generate(
        ckpt_path=gcfg["ckpt"],
        prompt=gcfg.get("prompt", ""),
        steps=int(gcfg.get("steps", 200)),
        temperature=float(gcfg.get("temperature", 1.0)),
        top_k=top_k,
        top_p=top_p,
        device=gcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    )
    print(txt)


if __name__ == "__main__":
    main()
