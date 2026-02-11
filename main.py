import argparse

import torch

from model.bigram import BigramLanguageModel
from model.config import ModelConfig
from trainer.trainer import Trainer
from loader.localfile import LocalFileModelLoader
from trainer.data_splitter import UniformDataSplitter
from tokenization.one_letter_tokenizer import OneLetterTokenizer


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
    args = parser.parse_args()

    block_size = 8  # Used in both ModelConfig and Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Reading and tokenizing data...")
    with open("data/quran-simple.txt", encoding="utf-8") as f:
        text = f.read()
    tokenizer = OneLetterTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    print("Encoding text...")
    encoded_data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Total tokens: {len(encoded_data)}")

    print("Splitting data...")
    train_data, val_data = UniformDataSplitter(encoded_data, split_ratio=0.9).split()
    print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

    # Load or create model
    loader = LocalFileModelLoader()
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
