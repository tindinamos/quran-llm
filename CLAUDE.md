# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A transformer-based language model for Quranic Arabic text generation. The model uses character-level tokenization with special handling for Arabic diacritical marks (tashkeel).

## Common Commands

### Setup
```bash
uv sync                    # Install dependencies (using uv package manager)
source .venv/bin/activate  # Activate virtual environment
```

### Training
```bash
python main.py                                    # Train from scratch (100 iterations)
python main.py --checkpoint checkpoints/model.pt  # Resume training from checkpoint
python main.py --save-checkpoint path/to/model.pt # Save to custom location
```

### Generation
```bash
python main.py --generate --checkpoint checkpoints/model.pt  # Generate 500 tokens
python main.py --generate --checkpoint checkpoints/model.pt --max-tokens 1000  # Custom length
python main.py --generate  # Generate from untrained model (random output)
```

### Inspection
```bash
python main.py --config --checkpoint checkpoints/model.pt  # View model config and architecture
```

### Code Quality
```bash
ruff check --fix .  # Lint and auto-fix
ruff format .       # Format code
mypy .              # Type checking (if needed)
```

### Testing
```bash
python tokenization/one_letter_tokenizer.py  # Test tokenizer
```

## Architecture

### Entry Point (`main.py`)
Three operational modes controlled by CLI flags:
- **Training mode** (default): Train or resume training, save checkpoint
- **Generation mode** (`--generate`): Load model and generate text
- **Config mode** (`--config`): Inspect checkpoint contents

### Model Architecture (`model/`)
- **BigramLanguageModel** (`bigram.py`): Main model class
  - Token and position embeddings
  - Sequential transformer blocks
  - Final layer norm + language model head
  - Stores config as `self.config` for checkpointing
- **Block** (`block.py`): Transformer block with pre-norm architecture
  - Multi-head self-attention with residual connection
  - Feed-forward network with residual connection
- **MultiHeadAttention** (`multi_head.py`): Parallel attention heads
- **Head** (`head.py`): Single self-attention head with causal masking
- **FeedForward** (`feed_forward.py`): Two-layer MLP with ReLU
- **ModelConfig** (`config.py`): Configuration dataclass holding hyperparameters

### Tokenization (`tokenization/`)
**OneLetterTokenizer** (`one_letter_tokenizer.py`): Custom character-level tokenizer
- Groups Arabic letters with their diacritics as single tokens
- Handles verse numbers (numerics followed by `|`)
- Vocabulary includes: Arabic letters, diacritics (ًٌٍَُِّْ), numerics, special chars (۩)
- ~654 tokens in vocabulary

### Training (`trainer/`)
- **Trainer** (`trainer.py`): Training loop with loss estimation
  - Creates AdamW optimizer internally during `run()`
  - Evaluates on train/val splits at intervals
  - Random batch sampling from data
- **UniformDataSplitter** (`data_splitter.py`): Splits data into train/val sets

### Persistence (`loader/`)
**LocalFileModelLoader** (`localfile.py`): Checkpoint management
- Saves: model state dict, optimizer state dict, config dict
- Creates directories automatically
- Returns optimizer state separately (caller creates optimizer after model)

## Key Patterns

### Checkpoint Structure
Checkpoints contain three components:
```python
{
    "model_state_dict": ...,      # Model weights
    "optimizer_state_dict": ...,  # Optimizer state (for resuming training)
    "config": model.config.__dict__  # Architecture config (vocab_size, n_embd, etc.)
}
```

### Loading Workflow
The model config must be reconstructed before creating the model:
1. Load checkpoint file
2. Reconstruct `ModelConfig` from saved dict
3. Create model skeleton using config
4. Load state dict into model
5. Create optimizer with model parameters
6. Load optimizer state dict

### Hyperparameters
Default hyperparameters are hardcoded in `main.py` (not in a config file):
- `n_embd=32`: Embedding dimension
- `n_head=2`: Number of attention heads
- `n_layer=2`: Number of transformer blocks
- `block_size=8`: Context window size
- `dropout=0.2`: Dropout probability
- `batch_size=4`: Training batch size
- `learning_rate=1e-3`: AdamW learning rate
- `max_iters=100`: Training iterations
- `eval_interval=10`: Steps between evaluations

### Device Handling
- Auto-detects CUDA availability
- Defaults to CPU (current setup)
- Device is passed to model and used for all tensors

## Data

- Training data: `data/quran-simple.txt` (UTF-8 encoded)
- Contains complete Quran text with verse numbering
- 90/10 train/val split
- No preprocessing required (tokenizer handles all text)

## Development Notes

- The project uses **uv** as the package manager (not pip)
- Python 3.13+ required
- Ruff is configured for code formatting and linting (see `pyproject.toml`)
- Line length: 88 characters
- No docstring requirements (docstring checks disabled)
- Checkpoints directory is gitignored
