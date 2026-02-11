# Quran Language Model

A transformer-based language model trained on Quran text for Arabic text generation.

## Overview

This project implements a small transformer model (bigram-style architecture with multi-head attention) trained on Quranic Arabic text. The model uses character-level tokenization with Arabic diacritics support.

**Model Architecture:**
- Transformer blocks with multi-head self-attention
- Feed-forward layers with ReLU activation
- Layer normalization
- Positional embeddings

**Features:**
- Custom tokenizer that groups Arabic letters with diacritical marks
- Model checkpointing and resumption
- Text generation from trained models
- Checkpoint inspection

## Setup

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd quran
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Activate virtual environment**
   ```bash
   source .venv/bin/activate
   ```

## Usage

### Training

#### Train from scratch

```bash
python main.py
```

This will:
- Load and tokenize the Quran text
- Create a new model with default hyperparameters
- Train for 100 iterations
- Save checkpoint to `checkpoints/model.pt`

#### Resume training from checkpoint

```bash
python main.py --checkpoint checkpoints/model.pt
```

Loads the model and optimizer state and continues training.

#### Custom checkpoint location

```bash
python main.py --save-checkpoint models/my_model.pt
```

### Text Generation

#### Generate from a trained model

```bash
python main.py --generate --checkpoint checkpoints/model.pt
```

Generates 500 tokens of text using the trained model.

#### Generate with custom length

```bash
python main.py --generate --checkpoint checkpoints/model.pt --max-tokens 1000
```

#### Generate from untrained model (random)

```bash
python main.py --generate
```

Creates a random model and generates text (useful for testing).

### Checkpoint Inspection

#### View checkpoint configuration

```bash
python main.py --config --checkpoint checkpoints/model.pt
```

Displays:
- Model configuration (vocab_size, n_embd, n_head, n_layer, etc.)
- First 10 model layers with shapes
- Total parameter count
- Whether optimizer state is available

Example output:
```
=== Checkpoint Information ===

Config:
  vocab_size: 654
  n_embd: 32
  n_head: 2
  n_layer: 2
  block_size: 8
  dropout: 0.2

Model state dict keys:
  token_embedding_table.weight: torch.Size([654, 32])
  position_embedding_table.weight: torch.Size([8, 32])
  ...

Total parameters: 68,302

Optimizer state available: True
```

## Command-Line Arguments

### Common Arguments

- `--checkpoint PATH`: Path to checkpoint file to load
- `--save-checkpoint PATH`: Where to save checkpoint (default: `checkpoints/model.pt`)

### Mode Selection

- `--generate`: Generate text instead of training
- `--config`: Show checkpoint configuration and exit

### Generation Options

- `--max-tokens N`: Number of tokens to generate (default: 500)

## Project Structure

```
quran/
├── model/              # Model architecture
│   ├── bigram.py      # Main model class
│   ├── head.py        # Self-attention head
│   ├── multi_head.py  # Multi-head attention
│   ├── block.py       # Transformer block
│   ├── feed_forward.py # Feed-forward layer
│   └── config.py      # Model configuration
├── trainer/            # Training logic
│   ├── trainer.py     # Training loop
│   └── data_splitter.py # Data splitting
├── tokenization/       # Tokenizer
│   └── one_letter_tokenizer.py # Character-level tokenizer with diacritics
├── loader/             # Model persistence
│   └── localfile.py   # Save/load checkpoints
├── data/               # Training data
│   └── quran-simple.txt # Quran text
├── checkpoints/        # Saved models (gitignored)
├── main.py            # Main entry point
└── pyproject.toml     # Dependencies
```

## Model Hyperparameters

Default configuration (editable in `main.py`):

- `vocab_size`: Determined by tokenizer (~654 tokens)
- `n_embd`: 32 (embedding dimension)
- `n_head`: 2 (number of attention heads)
- `n_layer`: 2 (number of transformer blocks)
- `block_size`: 8 (context window)
- `dropout`: 0.2
- `batch_size`: 4
- `learning_rate`: 1e-3
- `max_iters`: 100
- `eval_interval`: 10

## Examples

### Complete training workflow

```bash
# 1. Train a model
python main.py

# 2. Check the saved model
python main.py --config --checkpoint checkpoints/model.pt

# 3. Generate text
python main.py --generate --checkpoint checkpoints/model.pt

# 4. Resume training for more iterations
python main.py --checkpoint checkpoints/model.pt
```

### Quick generation test

```bash
# Generate from untrained model (will be gibberish)
python main.py --generate --max-tokens 100
```

## Development

### Code formatting

```bash
ruff check --fix .
ruff format .
```

### Running tests

The tokenizer includes a test mode:

```bash
python tokenization/one_letter_tokenizer.py
```

## Notes

- The model currently runs on CPU (CUDA support available but requires CUDA toolkit)
- Training data is the complete Quran text with verse numbering
- Checkpoints are saved with model weights, optimizer state, and configuration
- The tokenizer groups Arabic letters with diacritical marks as single tokens

## License

[Add your license here]
