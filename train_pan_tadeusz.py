import os
import random
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional tqdm progress bar with safe no-op fallback
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - environments without tqdm
    _tqdm = None


class _NoopBar:
    def __init__(self, iterable, total=None, desc=None):
        self._iterable = iterable
        self.total = total
        self.desc = desc

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, *_, **__):
        pass

    def update(self, *_):
        pass

    def close(self):
        pass


def make_pbar(iterable, total=None, desc=None):
    if _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc)
    return _NoopBar(iterable, total=total, desc=desc)


# Support both package and local runs
try:
    from .model import BDH_GPU
    from .tokenizer import SimpleTokenizer
    from .config_utils import load_config
except ImportError:
    from model import BDH_GPU
    from tokenizer import SimpleTokenizer
    from config_utils import load_config


@dataclass
class Batch:
    x: torch.Tensor  # (B, T)
    y: torch.Tensor  # (B, T)


class TokDataset(Dataset):
    def __init__(self, ids: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        self.data = ids

    def __len__(self):
        return max(0, self.data.numel() - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + 1 + self.seq_len]
        return x, y


def collate(examples):
    xs, ys = zip(*examples)
    return Batch(x=torch.stack(xs, dim=0), y=torch.stack(ys, dim=0))


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _optimizer_to(optimizer: torch.optim.Optimizer, device: str):
    """Move optimizer state tensors to device (after loading state_dict)."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def main():
    cfg = load_config()
    mcfg = cfg.get("model", {})
    tcfg = cfg.get("training", {})
    tokcfg = cfg.get("tokenizer", {})

    # Device resolve with fallback
    device_cfg = tcfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    else:
        device = device_cfg

    set_seed(int(tcfg.get("seed", 1337)))

    data_path = tcfg["data"]
    text = load_text(data_path)

    # Tokenizer: prefer loading from training.tokenizer_json; if missing, try training from tokenizer.files
    tokenizer_json_path = tcfg.get("tokenizer_json", "")
    tk = None
    if tokenizer_json_path and os.path.exists(tokenizer_json_path):
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tk = SimpleTokenizer.from_json_str(f.read())
    else:
        files = tokcfg.get("files", [])
        vocab_size = int(tokcfg.get("vocab_size", 2000))
        if files:
            tk = SimpleTokenizer.train_from_files(files, vocab_size=vocab_size)
        else:
            # fallback: train from the training data file
            tk = SimpleTokenizer.train_from_text(data_path, vocab_size=vocab_size)
        if tokenizer_json_path:
            os.makedirs(os.path.dirname(tokenizer_json_path), exist_ok=True)
            with open(tokenizer_json_path, "w", encoding="utf-8") as f:
                f.write(tk.to_json_str())

    ids = torch.tensor(tk.encode_ids(text, add_bos_eos=False), dtype=torch.long)
    dataset = TokDataset(ids, seq_len=int(tcfg.get("seq_len", 256)))
    loader = DataLoader(
        dataset,
        batch_size=int(tcfg.get("batch_size", 32)),
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )

    # Model
    model = BDH_GPU(
        vocab_size=tk.vocab_size,
        D=int(mcfg.get("D", 256)),
        H=int(mcfg.get("H", 4)),
        N=int(mcfg.get("N", 4096)),
        L=int(mcfg.get("L", 4)),
        dropout=float(mcfg.get("dropout", 0.1)),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 3e-4)))
    loss_fn = nn.CrossEntropyLoss()

    # --- Optional resume ---
    resume_cfg = tcfg.get("resume", "")  # can be falsey, true, or a path
    resume_path: str | None = None
    if isinstance(resume_cfg, bool):
        if resume_cfg:
            resume_path = tcfg.get(
                "save", os.path.join(os.path.dirname(__file__), "ckpt.pt")
            )
    elif isinstance(resume_cfg, str) and resume_cfg.strip():
        resume_path = resume_cfg

    ckpt = None
    global_step = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        try:
            model.load_state_dict(ckpt["model"], strict=True)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])  # type: ignore
                _optimizer_to(optimizer, device)
            global_step = int(ckpt.get("global_step", 0))
            print(
                f"[resume] Loaded checkpoint from {resume_path} at step {global_step}"
            )
        except Exception as e:
            print(f"[resume] Failed to load from {resume_path}: {e}")

    model.train()
    # Training schedule: prefer epochs if provided; otherwise, run for a fixed number of steps
    steps_cfg = int(tcfg.get("steps", 1000))
    epochs_cfg = int(tcfg.get("epochs", 0))
    steps_per_epoch = max(1, len(loader))  # avoid div-by-zero for tiny datasets

    if epochs_cfg > 0:
        total_steps = epochs_cfg * steps_per_epoch
        epochs = epochs_cfg
    else:
        total_steps = steps_cfg
        # derive epochs to give a sensible epoch counter in UI
        epochs = max(1, (steps_cfg + steps_per_epoch - 1) // steps_per_epoch)

    ema_loss = None
    ema_beta = 0.98  # smoothing for display

    # --- LR Scheduler (optional) ---
    scheduler = None
    sched_cfg = tcfg.get("scheduler", {}) or {}
    sched_type = str(sched_cfg.get("type", "none")).lower()
    if sched_type == "cosine":
        warmup = int(sched_cfg.get("warmup_steps", 0))
        min_lr_factor = float(sched_cfg.get("min_lr_factor", 0.1))

        def lr_lambda(step: int):
            s = max(step, 0)
            if warmup > 0 and s < warmup:
                return max(s, 1) / float(warmup)
            remain = max(total_steps - warmup, 1)
            progress = min(max((s - warmup) / remain, 0.0), 1.0)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine

        from torch.optim.lr_scheduler import LambdaLR

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=global_step - 1)
    elif sched_type == "step":
        step_size = int(sched_cfg.get("step_size", 1000))
        gamma = float(sched_cfg.get("gamma", 0.5))
        from torch.optim.lr_scheduler import StepLR

        scheduler = StepLR(
            optimizer, step_size=step_size, gamma=gamma, last_epoch=global_step - 1
        )

    # If resuming and we have a scheduler state, load it
    if (
        ckpt is not None
        and scheduler is not None
        and "scheduler" in ckpt
        and ckpt["scheduler"]
    ):
        try:
            scheduler.load_state_dict(ckpt["scheduler"])  # type: ignore
        except Exception as e:
            print(f"[resume] Failed to load scheduler state: {e}")

    for epoch in range(1, epochs + 1):
        # inner progress bar per epoch
        pbar = make_pbar(loader, total=steps_per_epoch, desc=f"epoch {epoch}/{epochs}")
        for batch in pbar:
            if epochs_cfg == 0 and global_step >= total_steps:
                break  # stop exactly at requested steps

            global_step += 1
            x = batch.x.to(device)
            y = batch.y.to(device)
            logits = model(x)  # (B, T, V)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # stats
            loss_item = float(loss.detach().item())
            ema_loss = (
                loss_item
                if ema_loss is None
                else (ema_beta * ema_loss + (1 - ema_beta) * loss_item)
            )
            lr = optimizer.param_groups[0].get("lr", None)
            # live bar update (noop if tqdm missing)
            # build lr string safely
            lr_str = f"{lr:.2e}" if isinstance(lr, (float, int)) else "-"
            try:
                pbar.set_postfix(
                    {
                        "step": f"{global_step}/{total_steps}",
                        "loss": f"{loss_item:.4f}",
                        "loss_ema": f"{ema_loss:.4f}",
                        "lr": lr_str,
                    }
                )
            except Exception:
                if global_step % 50 == 0:
                    print(
                        f"epoch {epoch}/{epochs} step {global_step}/{total_steps} "
                        f"loss {loss_item:.4f} (ema {ema_loss:.4f}) lr {lr_str}"
                    )

        # if we were limited by steps, we may already be done
        if epochs_cfg == 0 and global_step >= total_steps:
            break

    # Save
    save_path = tcfg.get("save", os.path.join(os.path.dirname(__file__), "ckpt.pt"))
    save_dir = os.path.dirname(save_path) or "."
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "global_step": global_step,
            "vocab_size": tk.vocab_size,
            "tokenizer_json": tk.to_json_str(),
            "config": {
                "D": int(mcfg.get("D", 256)),
                "H": int(mcfg.get("H", 4)),
                "N": int(mcfg.get("N", 4096)),
                "L": int(mcfg.get("L", 4)),
                "dropout": float(mcfg.get("dropout", 0.1)),
            },
        },
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
