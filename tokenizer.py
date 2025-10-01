from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
except Exception:
    Tokenizer = None  # type: ignore


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


@dataclass
class SimpleTokenizer:
    """Lekki wrapper na HF tokenizers (BPE)."""

    tk: Any

    @classmethod
    def train_from_text(
        cls, text_path: str, vocab_size: int = 2000
    ) -> "SimpleTokenizer":
        if Tokenizer is None:
            raise RuntimeError(
                "Pakiet 'tokenizers' nie jest zainstalowany. Zainstaluj: pip install tokenizers"
            )
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))  # type: ignore
        tokenizer.pre_tokenizer = Whitespace()  # type: ignore
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)  # type: ignore
        tokenizer.train(files=[text_path], trainer=trainer)  # type: ignore
        return cls(tokenizer)

    @classmethod
    def train_from_files(
        cls, files: list[str], vocab_size: int = 2000
    ) -> "SimpleTokenizer":
        if Tokenizer is None:
            raise RuntimeError(
                "Pakiet 'tokenizers' nie jest zainstalowany. Zainstaluj: pip install tokenizers"
            )
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))  # type: ignore
        tokenizer.pre_tokenizer = Whitespace()  # type: ignore
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)  # type: ignore
        tokenizer.train(files=files, trainer=trainer)  # type: ignore
        return cls(tokenizer)

    @classmethod
    def from_json_str(cls, data: str) -> "SimpleTokenizer":
        if Tokenizer is None:
            raise RuntimeError(
                "Pakiet 'tokenizers' nie jest zainstalowany. Zainstaluj: pip install tokenizers"
            )
        return cls(Tokenizer.from_str(data))  # type: ignore

    def to_json_str(self) -> str:
        return self.tk.to_str()  # type: ignore

    def encode_ids(self, text: str, add_bos_eos: bool = False) -> List[int]:
        ids = self.tk.encode(text).ids  # type: ignore
        if add_bos_eos:
            bos = self.tk.token_to_id("[BOS]")  # type: ignore
            eos = self.tk.token_to_id("[EOS]")  # type: ignore
            if bos is None or eos is None:
                return ids
            return [bos] + ids + [eos]
        return ids

    def decode_ids(self, ids: List[int]) -> str:
        return self.tk.decode(ids)  # type: ignore

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()  # type: ignore
