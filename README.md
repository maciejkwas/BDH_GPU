## Dragon (eksperymentalny) – PL

Eksperymentalna architektura „Transformer‑like”: multi‑head, projekcja do bardzo szerokiej przestrzeni neuronowej N, powtarzanie tej samej warstwy (weight tying). Uwaga używa RoPE i maski przyczynowej.

### Wymagania

- Python 3.10+
- PyTorch (CPU lub CUDA)
- Opcjonalnie: `tqdm` (pasek postępu w konsoli)

### Szybki start

- Budowa tokenizera (na podstawie ścieżek z `tokenizer.files` w `config.json`):

```bash
python -m build_tokenizer
```

- Trening (wykorzystuje ustawienia z `training.*` w `config.json`):

```bash
python -m train_pan_tadeusz
```

- Generowanie z checkpointu:

```bash
python -m generate
```

- Inspekcja (saliency + eksport grafu):

```bash
python -m model_inspect
```

Podczas treningu w konsoli pojawia się pasek postępu (tqdm) z metrykami: `epoch`, `step`, `loss`, `loss_ema`, `lr`. Bez `tqdm` pojawią się okresowe logi tekstowe.

### Konfiguracja

Plik: `config.json`

- `tokenizer.files`, `tokenizer.vocab_size`, `tokenizer.out`
- `training.data`, `seq_len`, `batch_size`, `steps` lub `epochs`, `lr`, `device`, `save`, `tokenizer_json`
- `model.D`, `H`, `N`, `L`, `dropout`
- Sekcje `generate` i `inspect` wskazują odpowiednio checkpoint i parametry uruchomienia.

### Najważniejsze cechy

- RoPE i maska przyczynowa: szkic miał pseudokod, dodana realna implementacja RoPE i causal mask.
- Precyzyjne mnożenia per‑head: zamiast niejawnego broadcastu użyte `einsum("b t d, h d n -> b h t n")` dla czytelności i bezpieczeństwa wymiarów..
- Debug/interpretowalność: maski na `encoder`, `decoder_x`, `decoder_y` + metody `ablate_neurons` i `clear_masks`. `forward(..., return_debug=True)` może zwracać aktywacje pre‑ReLU do analizy.
- Zachowany styl: weight tying, residual, LayerNorm bez affine.
- Tokenizacja: przejście z poziomu znaków na BPE (większa semantyka, krótsze sekwencje).

### Podgląd grafu

Szybki podgląd `graph.json` z użyciem Pythona:

```bash
python -m view_graph --graph graph.json --out graph.png --html graph.html
```

Wymaga `networkx` i `matplotlib`; opcjonalnie `pyvis` do pliku HTML.

---

## Dragon (experimental) – EN

Experimental, Transformer‑like architecture: multi‑head, projection into a very wide “neuron space” N, repeated single tied layer (weight tying). Attention uses RoPE and a causal mask.

### Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- Optional: `tqdm` (console progress bar)

### Quickstart

- Build tokenizer (from `tokenizer.files` in `config.json`):

```bash
python -m build_tokenizer
```

- Train (uses `training.*` in `config.json`):

```bash
python -m train_pan_tadeusz
```

- Generate from checkpoint:

```bash
python -m generate
```

- Inspect (saliency + graph export):

```bash
python -m model_inspect
```

During training, a tqdm progress bar shows `epoch`, `step`, `loss`, `loss_ema`, and `lr`. Without `tqdm`, periodic textual logs are printed.

### Configuration

File: `config.json`

- `tokenizer.files`, `vocab_size`, `out`
- `training.data`, `seq_len`, `batch_size`, `steps` or `epochs`, `lr`, `device`, `save`, `tokenizer_json`
- `model.D`, `H`, `N`, `L`, `dropout`
- `generate` and `inspect` sections define checkpoint paths and run parameters.

### Key features/changes

- RoPE and causal mask implemented (the sketch had pseudocode; now fully functional).
- Per‑head einsum multiplication: `einsum("b t d, h d n -> b h t n")` instead of implicit broadcasting for clarity and shape safety.
- Debug/interpretability: masks for `encoder`, `decoder_x`, `decoder_y` + `ablate_neurons`, `clear_masks`. `forward(..., return_debug=True)` can return pre‑ReLU activations.
- Preserved style: weight tying, residual connections, LayerNorm without affine.
- Tokenization switched from characters to BPE for better semantics and shorter sequences.

### Graph preview

Preview `graph.json` using Python tools:

```bash
python -m view_graph --graph graph.json --out graph.png --html graph.html
```

Requires `networkx` and `matplotlib`; optional `pyvis` for interactive HTML.
