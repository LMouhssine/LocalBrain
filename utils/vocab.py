from __future__ import annotations

from dataclasses import dataclass


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def decode(self, token_id: int) -> str:
        if 0 <= token_id < len(self.id_to_token):
            return self.id_to_token[token_id]
        return UNK_TOKEN


def build_vocab(tokens: list[str], *, min_freq: int = 1) -> Vocab:
    counts: dict[str, int] = {}
    for token in tokens:
        token = (token or "").strip()
        if not token:
            continue
        counts[token] = counts.get(token, 0) + 1

    id_to_token = [PAD_TOKEN, UNK_TOKEN]
    for token, freq in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq < min_freq:
            continue
        id_to_token.append(token)

    token_to_id = {t: i for i, t in enumerate(id_to_token)}
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)


def fixed_vocab(tokens_in_order: list[str]) -> Vocab:
    id_to_token = [PAD_TOKEN, UNK_TOKEN] + tokens_in_order
    token_to_id = {t: i for i, t in enumerate(id_to_token)}
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token)

