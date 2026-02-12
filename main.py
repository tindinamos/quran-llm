import argparse

import torch

from model.bigram import BigramLanguageModel
from model.config import ModelConfig
from trainer.trainer import Trainer
from loader.localfile import LocalFileModelLoader
from trainer.data_splitter import UniformDataSplitter
from tokenization.one_letter_tokenizer import OneLetterTokenizer
from tokenization.byte_pair_encoding_tokenizer import BytePairEncodingTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train or load a Quran language model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file to load and resume training",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default="checkpoints/model.pt",
        help="Path to save checkpoint after training (default: checkpoints/model.pt)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate text instead of training",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate (default: 500)",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Print checkpoint configuration and exit (requires --checkpoint)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["oneletter", "bpe"],
        default="oneletter",
        help="Tokenizer to use: oneletter (character-level with diacritics) or bpe (byte-pair encoding)",
    )
    args = parser.parse_args()

    # CONFIG MODE
    if args.config:
        if not args.checkpoint:
            print("Error: --config requires --checkpoint to be specified")
            return

        print(f"Loading checkpoint info from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

        print("\n=== Checkpoint Information ===\n")
        print("Config:")
        for key, value in checkpoint["config"].items():
            print(f"  {key}: {value}")

        print("\nModel state dict keys:")
        for key in list(checkpoint["model_state_dict"].keys())[:10]:
            shape = checkpoint["model_state_dict"][key].shape
            print(f"  {key}: {shape}")
        if len(checkpoint["model_state_dict"]) > 10:
            print(f"  ... and {len(checkpoint['model_state_dict']) - 10} more layers")

        total_params = sum(
            p.numel() for p in checkpoint["model_state_dict"].values()
        )
        print(f"\nTotal parameters: {total_params:,}")

        print(f"\nOptimizer state available: {'optimizer_state_dict' in checkpoint}")

        return

    block_size = 8  # Used in both ModelConfig and Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Reading and tokenizing data...")
    with open("data/quran-simple.txt", encoding="utf-8") as f:
        text = f.read()

    # Initialize tokenizer based on argument
    if args.tokenizer == "oneletter":
        print("Using OneLetterTokenizer...")
        tokenizer = OneLetterTokenizer(text)
    elif args.tokenizer == "bpe":
        print("Using BytePairEncodingTokenizer (building vocabulary)...")
        tokenizer = BytePairEncodingTokenizer(text)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load or create model
    loader = LocalFileModelLoader()

    # GENERATION MODE
    if args.generate:
        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}...")
            model, _, config = loader.load_model(args.checkpoint, device)
        else:
            print("Creating random model for generation...")
            model = BigramLanguageModel(
                ModelConfig(
                    vocab_size=tokenizer.vocab_size,
                    n_embd=32,
                    n_head=2,
                    n_layer=2,
                    block_size=block_size,
                    dropout=0.2,
                )
            ).to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Generate text
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(f"\nGenerating {args.max_tokens} tokens...\n")
        generated = model.generate(context, max_new_tokens=args.max_tokens)
        print(tokenizer.decode(generated[0].tolist()))
        return

    # TRAINING MODE
    print("Encoding text...")
    encoded_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Total tokens: {len(encoded_data)}")

    print("Splitting data...")
    train_data, val_data = UniformDataSplitter(encoded_data, split_ratio=0.9).split()
    print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        model, optimizer_state, config = loader.load_model(args.checkpoint, device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create optimizer AFTER model is loaded, then load optimizer state
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(optimizer_state)
        print("Optimizer state loaded.")
    else:
        print("Creating new model...")
        model = BigramLanguageModel(
            ModelConfig(
                vocab_size=tokenizer.vocab_size,
                n_embd=32,
                n_head=2,
                n_layer=2,
                block_size=block_size,
                dropout=0.2,
            )
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create new optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create trainer and train
    print("Starting training...")
    trainer = Trainer(
        model=model,
        device=device,
        eval_iters=30,
        block_size=block_size,
        batch_size=4,
    )
    trainer.run({"train": train_data, "val": val_data}, 100, 10, 1e-3)

    # Save checkpoint
    print(f"\nSaving checkpoint to {args.save_checkpoint}...")
    loader.save_model(model, optimizer, args.save_checkpoint)

    print("Training complete!")


if __name__ == "__main__":
    main()
