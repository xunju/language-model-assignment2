"""数据准备和字符词表。作业允许 BPE，但不强制；此处不依赖外部分词库。"""
import hashlib
import json
from pathlib import Path
import urllib.request

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tinyshakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharTokenizer:
    """固定字符集合，每个字符一一映射到一个 token，不引入 <UNK>。"""

    def __init__(self, characters):
        self.characters = list(characters)
        self.stoi = {c: i for i, c in enumerate(self.characters)}

    def encode(self, text):
        try:
            return [self.stoi[c] for c in text]
        except KeyError as exc:
            raise ValueError(f"提示词包含词表外字符：{exc.args[0]!r}") from exc

    def decode(self, indices):
        return "".join(self.characters[int(i)] for i in indices)


def prepare_data():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        # 标准下载；若运行环境代理失效，可由使用者手动下载到 DATA_PATH。
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            DATA_PATH.write_bytes(response.read())
    raw = DATA_PATH.read_bytes()
    text = raw.decode("utf-8")
    boundary = int(0.9 * len(text))
    train_text, val_text = text[:boundary], text[boundary:]
    # 仅从训练段构建词表，避免利用验证数据决定词表。
    tokenizer = CharTokenizer(sorted(set(train_text)))
    train = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    validation = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    metadata = {"url": DATA_URL, "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw), "characters": len(text),
                "train_tokens": len(train), "validation_tokens": len(validation),
                "split": "contiguous first 90% / last 10%; windows never cross split",
                "vocab_size": len(tokenizer.characters), "tokenizer": "training-only character vocabulary",
                "characters_vocabulary": tokenizer.characters}
    (ROOT / "data" / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return tokenizer, train, validation, metadata


def sample_batch(tokens, batch_size, context_length, generator, device):
    # length=T+1，输入 [0:T]，目标 [1:T+1]，明确对应下一个字符。
    starts = torch.randint(len(tokens) - context_length, (batch_size,), generator=generator)
    offsets = torch.arange(context_length + 1)
    chunks = tokens[starts[:, None] + offsets[None, :]]
    return chunks[:, :-1].contiguous().to(device), chunks[:, 1:].contiguous().to(device)


if __name__ == "__main__":
    _, _, _, info = prepare_data()
    print(json.dumps(info, ensure_ascii=False, indent=2))
