import json
from typing import List

import torch

from .model import BDH_GPU
from .tokenizer import SimpleTokenizer
from .config_utils import load_config


@torch.no_grad()
def load_model(ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = BDH_GPU(
        vocab_size=ckpt["vocab_size"],
        D=cfg["D"],
        H=cfg["H"],
        N=cfg["N"],
        L=cfg["L"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    tk = None
    if "tokenizer_json" in ckpt:
        tk = SimpleTokenizer.from_json_str(ckpt["tokenizer_json"])
    return model.eval(), tk


@torch.no_grad()
def neuron_salience(
    model: BDH_GPU, x: torch.Tensor, max_neurons: int = 2000
) -> List[float]:
    # Prosty wskaźnik: spadek średniego logproba następnego tokena po ablacji neuronu
    B, T = x.shape
    logits_base = model(x)  # (B, T, V)
    probs_base = torch.log_softmax(logits_base[:, :-1, :], dim=-1)
    tgt = x[:, 1:]
    base_lp = probs_base.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()

    N = model.N
    n_eval = min(N, max_neurons)
    scores = torch.zeros(n_eval, device=x.device)
    for i in range(n_eval):
        model.clear_masks()
        model.ablate_neurons([i], scale=0.0)
        logits = model(x)
        probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        lp = probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
        scores[i] = (base_lp - lp).item()
    model.clear_masks()
    return scores.tolist()


@torch.no_grad()
def export_graph(model: BDH_GPU, top_k: int = 5):
    # Bardzo uproszczony graf: dla każdego neuronu wybierz top-k tokenów z readout przez encoder
    # Przybliżenie: wpływ neuronu i to wiersz encodera[i,:] przeniesiony na logity D@readout
    E = model.encoder  # (N, D)
    R = model.readout  # (D, V)
    W = E @ R  # (N, V)
    V = W.size(1)
    nodes = [{"id": f"n{i}", "type": "neuron"} for i in range(model.N)]
    edges = []
    top_vals, top_idx = torch.topk(W, k=min(top_k, V), dim=1)
    for i in range(model.N):
        for val, j in zip(top_vals[i].tolist(), top_idx[i].tolist()):
            edges.append({"source": f"n{i}", "target": f"t{j}", "weight": float(val)})
    tokens = [{"id": f"t{j}", "type": "token", "idx": j} for j in range(V)]
    return {"nodes": nodes + tokens, "edges": edges}


def main():
    cfg = load_config()
    icfg = cfg.get("inspect", {})
    device = icfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt = icfg["ckpt"]
    text = icfg.get("text", "Litwo! Ojczyzno moja!")
    seq_len = int(icfg.get("seq_len", 128))
    graph_out = icfg.get("graph_out", "graph.json")
    salience_top = int(icfg.get("salience_top", 50))

    model, tk = load_model(ckpt, device)
    if tk is not None:
        ids = tk.encode_ids(text)
    else:
        # fallback char encoding if old checkpoint
        stoi = torch.load(ckpt, map_location="cpu").get("stoi", {})
        ids = [stoi.get(ch, 0) for ch in text]
    x = torch.tensor(ids[:seq_len], dtype=torch.long, device=device).unsqueeze(0)

    # Salience
    sal = neuron_salience(model, x, max_neurons=min(model.N, 1000))
    top_idx = sorted(range(len(sal)), key=lambda i: sal[i], reverse=True)[:salience_top]
    print("Top neurons:", [(i, round(sal[i], 4)) for i in top_idx])

    # Graph
    g = export_graph(model)
    with open(graph_out, "w", encoding="utf-8") as f:
        json.dump(g, f)
    print("Saved graph to", graph_out)


if __name__ == "__main__":
    main()
