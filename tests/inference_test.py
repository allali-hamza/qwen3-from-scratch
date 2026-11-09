import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from qwen3_from_scratch.generation import generate
from qwen3_from_scratch.layers.decoder import DecoderBlock
from qwen3_from_scratch.model import CausalLanguageModel, TransformerBackbone
from qwen3_from_scratch.weights import load_pretrained_weights

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    prompts = [
        "1 + 1 = ",
        "La capitale du Japon est",
        "Écris une phrase sur les océans :",
        "En Python, une liste est",
    ]

    decoders = nn.ModuleList(
        [
            DecoderBlock(
                hidden_size=2048,
                h_q=16,
                h_kv=8,
                teta_base=1_000_000.0,
                d_h=128,
                Dff=6144,
            )
            for _ in range(28)
        ]
    )
    backbone = TransformerBackbone(
        hidden_size=2048,
        vocab_size=151_936,
        decoders=decoders,
    )

    checkpoint_dir = snapshot_download(
        "Qwen/Qwen3-1.7B",
        local_files_only=True,
        allow_patterns=["*.safetensors"],
    )
    model = CausalLanguageModel(backbone)
    load_pretrained_weights(model, checkpoint_dir)
    model = model.to(device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print(f"Réponse: {generate(model, tokenizer, prompt, max_new_tokens=100)}")


if __name__ == "__main__":
    main()
