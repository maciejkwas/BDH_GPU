"""
Moduł eksperymentalny: hybryda uwagi i projekcji do dużej przestrzeni „neuronalnej”.

Strukturalnie przypomina Transformera z podziałem na głowice, ale:
- latent (D) jest rzutowany do bardzo szerokiej przestrzeni N przez dekodery per‑head,
- używa powtarzanych (weight‑tied) warstw,
- uwaga wykorzystuje RoPE i maskę przyczynową.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn


class BDH_GPU(nn.Module):
    """Minimalna, działająca implementacja modelu z pliku szkicowego.

    Parametry
    - vocab_size: rozmiar słownika (liczba tokenów)
    - D: wymiar wewnętrzny (latent)
    - H: liczba głowic
    - N: rozmiar „przestrzeni neuronowej” (musi być podzielne przez H)
    - L: liczba powtórzeń warstwy (weight tying)
    - dropout: p(dropout)
    """

    def __init__(
        self,
        vocab_size: int,
        D: int = 256,
        H: int = 4,
        N: int = 32768,
        L: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert N % H == 0, "N musi być podzielne przez H"

        self.D, self.H, self.N, self.L = D, H, N, L

        # Warstwa wejściowa - embedding tokenów
        self.wte = nn.Embedding(vocab_size, D)

        # Główne macierze parametrów
        # E: (N x D) – enkoder z przestrzeni neuronowej do latentu
        self.encoder = nn.Parameter(torch.empty((N, D)).normal_(std=0.02))

        # Dekodery per-head: (H, D, N/H)
        self.decoder_x = nn.Parameter(torch.empty((H, D, N // H)).normal_(std=0.02))
        self.decoder_y = nn.Parameter(torch.empty((H, D, N // H)).normal_(std=0.02))

        # Warstwa wyjściowa do predykcji tokenów (równoważnik Linear(D, vocab_size, bias=False))
        self.readout = nn.Parameter(torch.empty((D, vocab_size)).normal_(std=0.02))

        # Uwaga z RoPE i maską przyczynową
        self.attn = LinearAttention()

        # Normalizacja i dropout
        self.ln = nn.LayerNorm(D, elementwise_affine=False)
        self.drop = nn.Dropout(dropout)

        # Opcjonalne maski do debug/ablacji (jako Parameters bez gradientu)
        self.mask_encoder = nn.Parameter(torch.ones((N, D)), requires_grad=False)
        self.mask_decoder_x = nn.Parameter(
            torch.ones((H, D, N // H)), requires_grad=False
        )
        self.mask_decoder_y = nn.Parameter(
            torch.ones((H, D, N // H)), requires_grad=False
        )

    # --- API debug/interpret ---
    def clear_masks(self):
        with torch.no_grad():
            self.mask_encoder.fill_(1.0)
            self.mask_decoder_x.fill_(1.0)
            self.mask_decoder_y.fill_(1.0)

    def ablate_neurons(
        self, indices: list[int], heads: list[int] | None = None, scale: float = 0.0
    ):
        """Wyzeruj (lub przeskaluj) wybrane neurony N w dekoderach i enkoderze.

        indices: indeksy w [0, N)
        heads: jeśli podane, zastosuj tylko do tych głowic (dla dekoderów)
        scale: skala maski (0.0 = pełna ablacja)
        """
        N = self.N
        n_per_head = N // self.H
        if heads is None:
            heads = list(range(self.H))
        with torch.no_grad():
            for i in indices:
                h = i // n_per_head
                j = i % n_per_head
                if h in heads:
                    self.mask_decoder_x.data[h, :, j] = scale
                    self.mask_decoder_y.data[h, :, j] = scale
                # encoder jest wspólny: rząd i opisuje neuron i
                self.mask_encoder.data[i, :] = scale

    def forward(self, idx: torch.Tensor, return_debug: bool = False):
        """Zwraca logity (B, T, V). Jeśli return_debug=True, także słownik akt.*"""
        B, T = idx.shape

        # 1) Embedding + wstępna normalizacja
        v_ast = self.ln(self.wte(idx).unsqueeze(1))  # (B, 1, T, D)
        debug = {}

        # 2) Powtarzane warstwy z dzieleniem wag
        for _ in range(self.L):
            # 3) „x” – projekcja do przestrzeni neuronowej per-head
            # Użyj einsum, by jawnie zsumować po D i uzyskać (B, H, T, N/H)
            dec_x = self.decoder_x * self.mask_decoder_x
            x_lin = torch.einsum("b t d, h d n -> b h t n", v_ast.squeeze(1), dec_x)
            x = F.relu(x_lin)
            if return_debug:
                debug.setdefault("x_pre_relu", []).append(x_lin.detach())

            # 4) Uwaga (zwraca (B, 1, T, D))
            a_ast = self.attn(Q=x, K=x, V=v_ast)

            # 5) „y” – kolejna projekcja i gating przez x
            # (B, 1, T, D) x (H, D, N/H) -> (B, H, T, N/H)
            dec_y = self.decoder_y * self.mask_decoder_y
            y_lin = torch.einsum(
                "b t d, h d n -> b h t n", self.ln(a_ast).squeeze(1), dec_y
            )
            y_proj = y_lin
            y = F.relu(y_proj) * x
            if return_debug:
                debug.setdefault("y_pre_relu", []).append(y_lin.detach())

            # 6) Powrót do pełnego N i update latentu
            y = y.transpose(1, 2).reshape(B, 1, T, self.N)  # (B, 1, T, N)
            y = self.drop(y)

            # (B, 1, T, N) @ (N, D) -> (B, 1, T, D)
            enc = self.encoder * self.mask_encoder
            v_ast = v_ast + self.ln(y @ enc)
            v_ast = self.ln(v_ast)

        # 7) Logity tokenów: (B, 1, T, D) @ (D, V) -> (B, 1, T, V)
        logits = v_ast.squeeze(1) @ self.readout  # (B, T, V)
        if return_debug:
            return logits, debug
        return logits


class LinearAttention(nn.Module):
    """Uwaga z RoPE i maską przyczynową; agreguje głowice przez średnią.

    Wejścia:
    - Q, K: (B, H, T, d_head)
    - V: (B, 1, T, D)

    Wyjście:
    - (B, 1, T, D)
    """

    def __init__(self, rope_base: float = 10000.0) -> None:
        super().__init__()
        self.rope_base = rope_base

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # [.., 0::2] and [.., 1::2]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # (-x2, x1) interleaved back
        out = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return out

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Zastosuj Rotary Positional Embedding do ostatniego wymiaru x.
        x: (B, H, T, d)
        """
        d = x.size(-1)
        # jeśli d jest nieparzyste, obetnij ostatni kanał do parzystej liczby
        if d % 2 == 1:
            x_main = x[..., : d - 1]
        else:
            x_main = x

        d_main = x_main.size(-1)
        # inv_freq: (d/2,)
        inv_freq = 1.0 / (
            self.rope_base
            ** (torch.arange(0, d_main, 2, device=x.device, dtype=x.dtype) / d_main)
        )
        T = x_main.size(-2)
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("t,f->tf", t, inv_freq)  # (T, d/2)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (T, d)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # (T, d)

        # Dopasuj do (B, H, T, d)
        while cos.dim() < x_main.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        cos = cos.expand_as(x_main)
        sin = sin.expand_as(x_main)

        x_rot = (x_main * cos) + (self._rotate_half(x_main) * sin)

        if d % 2 == 1:
            # doklej obcięty kanał bez rotacji
            pad = x[..., -1:].clone()
            x_rot = torch.cat([x_rot, pad], dim=-1)

        return x_rot

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        B, H, T, d_head = Q.shape
        # RoPE na Q, K
        Qr = self._apply_rope(Q)
        Kr = self._apply_rope(K)

        # Skalarowanie jak w Transformerze
        scale = 1.0 / math.sqrt(max(d_head, 1))
        attn_scores = torch.matmul(Qr, Kr.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Maska przyczynowa: pozwól na przeszłość i bieżący token (diagonal=0)
        # Zapobiega wierszom zawierającym same -inf (co prowadzi do NaN w softmax).
        causal = torch.tril(
            torch.ones((T, T), device=attn_scores.device, dtype=torch.bool), diagonal=0
        )
        attn_scores = attn_scores.masked_fill(~causal, float("-inf"))

        attn = torch.softmax(attn_scores, dim=-1)  # (B, H, T, T)

        # Rozszerz V na głowice i policz kontekst
        Vh = V.expand(B, H, T, V.size(-1))  # (B, H, T, D)
        out_h = torch.matmul(attn, Vh)  # (B, H, T, D)

        # Agreguj głowice do kształtu (B, 1, T, D)
        out = out_h.mean(dim=1, keepdim=True)
        return out
