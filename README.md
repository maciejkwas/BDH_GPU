# Dragon model (experimental)

Szkic architektury przypominającej Transformera z podziałem na głowice, ale z rzutowaniem do bardzo szerokiej „przestrzeni neuronowej” N oraz powtarzaniem warstw (weight tying). Uwaga wykorzystuje Rotary Positional Embeddings (RoPE) i maskę przyczynową.

## Trening na Pan Tadeuszu (poziom znaków)

1. Przygotuj plik `pan_tadeusz.txt` (UTF‑8).

2. Uruchom trening (na CPU/GPU):

```bash
python -m train_pan_tadeusz
```

Parametry ważne dla zasobów:

- `N` (domyślnie 4096 w skrypcie – zmniejszone względem szkicu 32768),
- `D`, `H`, `L`, `batch_size`, `seq_len`.

Checkpoint zapisywany jest do `ckpt_pan_tadeusz.pt`.

## Szybki smoke test w Pythonie

```python
import torch
from dragon.model import BDH_GPU

m = BDH_GPU(vocab_size=100, D=64, H=2, N=1024, L=2)
x = torch.randint(0, 100, (4, 128))
logits = m(x)
print(logits.shape)  # (4, 128, 100)
```

## Generowanie tekstu z checkpointu

Po treningu uruchom generator znakowy:

```bash
python -m generate \
	--ckpt ckpt_pan_tadeusz.pt \
	--prompt "Litwo! Ojczyzno moja!" \
	--steps 400 --temperature 0.9 --top_p 0.9
```
