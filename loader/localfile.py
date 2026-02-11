import os
from pathlib import Path

import torch

from model.bigram import BigramLanguageModel
from model.config import ModelConfig


class LocalFileModelLoader:
    """
    A simple model loader that saves and loads from the local filesystem.
    This is the most basic way to persist your model, and it's what we'll use in this project.
    """

    def save_model(self, model, optimizer, filepath):
        """
        Saves the model state AND the configuration used to build it.
        """
        print(f"Saving model to {filepath}...")

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # We create a dictionary bundle
        checkpoint = {
            # 1. The Matrix Weights
            "model_state_dict": model.state_dict(),
            # 2. The Optimizer State (needed to resume training)
            "optimizer_state_dict": optimizer.state_dict(),
            # 3. CRITICAL: The Architecture Config
            # We need this to know how big the matrices are when loading!
            "config": model.config.__dict__,
        }

        torch.save(checkpoint, filepath)
        print("Saved successfully.")

    def load_model(self, filepath, device="cpu"):
        """
        Reads the config, builds the skeleton, then loads the weights.
        Returns model and optimizer_state separately so optimizer can be created AFTER model.
        """
        print(f"Loading model from {filepath}...")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")

        # 1. Load the file
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        # 2. Reconstruct the Config object
        conf_dict = checkpoint["config"]
        config = ModelConfig(**conf_dict)

        # 3. Build the empty model skeleton using that config
        model = BigramLanguageModel(config)

        # 4. Pour the weights into the skeleton
        model.load_state_dict(checkpoint["model_state_dict"])

        # 5. Move to device
        model.to(device)
        print("Model loaded successfully.")

        # 6. Return optimizer state separately
        # The caller will create optimizer AFTER getting the model
        optimizer_state = checkpoint["optimizer_state_dict"]

        return model, optimizer_state, config
