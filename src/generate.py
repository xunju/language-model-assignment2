"""从本实验的最终 checkpoint 重现小模型采样；不依赖训练脚本的状态。"""
import argparse
import torch
from .data import ROOT, CharTokenizer
from .model import ModelConfig, DecoderLM
from .train import seed_everything, precision_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "results" / "small_final.pt"))
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--tokens", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(42)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = CharTokenizer(checkpoint["characters"])
    model = DecoderLM(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode(), precision_context(device):
        generated = model.generate(prompt, args.tokens, args.temperature, args.top_k, rng)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
