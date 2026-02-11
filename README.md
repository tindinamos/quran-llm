# Name Transcription

A Python project for Quran Tokenization and Generation

## Table of Contents

- [Features](#features)
- [Local Setup](#local-setup)
- [Google Colab Setup](#google-colab-setup)
- [Usage](#usage)
- [Available Models](#available-models)
- [Examples](#examples)

## Features

- Support for multiple AI models (GPT and Gemini)
- Batch processing of names
- Model comparison capabilities
- Few-shot learning support
- Evaluation metrics (WER, CER)
- CSV input/output support

## Local Setup

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Installation with uv (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd quran
   ```

2. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

   This will:
   - Create a virtual environment in `.venv`
   - Install all required packages
   - Install the project in editable mode

4. **Set up API keys**

   Create a `.env` file in the project root or export environment variables:
   ```bash
   # For GPT models
   export OPENAI_API_KEY="your-openai-api-key"

   # For Gemini models
   export GOOGLE_API_KEY="your-google-api-key"
   ```

5. **Run the project**
   ```bash
   # Using uv (automatically uses the virtual environment)
   uv run python models/main.py --help

   # Or activate the virtual environment manually
   source .venv/bin/activate
   python models/main.py --help
   ```

### Installation with pip

If you prefer using pip:

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd quran
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Set up API keys** (same as above)

## Google Colab Setup

To use this project in Google Colab:

### 1. Clone the Repository

```python
# Clone the repository
!git clone <repository-url>
%cd quran
```

### 2. Install Dependencies

```python
# Install the project and all dependencies
!pip install -e .
```

### 3. Set API Keys

```python
# Option 1: Using Colab Secrets (Recommended)
from google.colab import userdata
import os

# Set API keys from Colab secrets
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')

# To add secrets: Click the key icon in the left sidebar
```

```python
# Option 2: Direct input (Less secure - don't share the notebook!)
import os
import getpass

os.environ['OPENAI_API_KEY'] = getpass.getpass('Enter OpenAI API Key: ')
os.environ['GOOGLE_API_KEY'] = getpass.getpass('Enter Google API Key: ')
```

### 4. Import and Use

```python
# Now you can import and use the modules
from models.setup_engine import setup_engine

# Example: Set up GPT engine
transliterator, runner, evaluator = setup_engine("gpt", "4o-mini")

# Transliterate a single name
runner.run_single_name("محمد")
```

### Complete Colab Example

```python
# 1. Install
!git clone <repository-url>
%cd quran
!pip install -e .

# 2. Set up API keys
import os
from google.colab import userdata
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')

# 3. Use the model
from models.setup_engine import setup_engine

transliterator, runner, evaluator = setup_engine("gpt", "4o-mini")
runner.run_single_name("أحمد")
```

## Usage

### Basic Commands

#### Transliterate a single name

```bash
uv run python models/main.py --engine gpt --model 4o-mini --name "محمد"
```

#### Run on a test set

```bash
uv run python models/main.py --engine gpt --model 4o-mini --test-set simple --batch-size 4
```

#### Run on a custom CSV file

```bash
uv run python models/main.py --engine gemini --model 2.5-flash --csv path/to/file.csv --batch-size 8
```

#### Compare different batch sizes

```bash
uv run python models/main.py --engine gpt --model 4o-mini --test-set simple --compare 1 4 8 16
```

#### Compare different models

```bash
uv run python models/main.py --compare-models "gpt 4o-mini" "gemini 2.5-flash" --test-set simple
```

#### Use few-shot examples

```bash
uv run python models/main.py --engine gpt --model 4o-mini --name "محمد" --use-fewshots CREATED_SHORT
```

### Command Line Arguments

- `--engine`: AI engine to use (`gpt` or `gemini`)
- `--model`: Model version (see [Available Models](#available-models))
- `--name`: Single Arabic name to transliterate
- `--test-set`: Test set name (`simple`, `big`, `sample_test`, `nllb`, `mbart`, `nile`)
- `--csv`: Path to CSV file with names
- `--batch-size`: Number of names to process in one batch (default: 4)
- `--compare`: List of batch sizes to compare
- `--compare-models`: Compare multiple models
- `--save-dir`: Directory to save results
- `--use-fewshots`: Few-shot examples key
- `--debugging`: Logging level (default: `output`)
  - `detailed`: Show all logs (batch processing, evaluation metrics, progress bars)
  - `output`: Show only final results and progress bars
  - `none`: Silent mode (no output)

## Available Models

### GPT Models (OpenAI)

- `4o-mini`: GPT-4o Mini (default settings)
- `5-mini`: GPT-5 Mini
- `3.5-turbo`: GPT-3.5 Turbo

### Gemini Models (Google)

- `2.5-flash`: Gemini 2.5 Flash (fast, balanced)
- `2.5-flash-lite`: Gemini 2.5 Flash Lite (fastest)
- `2.5-pro`: Gemini 2.5 Pro (most capable)

### Test Sets

- `simple`: Basic test set
- `big`: Large test set
- `sample_test`: Sample test data
- `nllb`: NLLB model test data
- `mbart`: mBART model test data
- `nile`: NILE model test data

## Examples

### Example 1: Quick Single Name Test

```bash
uv run python models/main.py --engine gpt --model 4o-mini --name "فاطمة"
```

### Example 2: Batch Processing with Results Saving

```bash
uv run python models/main.py \
  --engine gemini \
  --model 2.5-flash \
  --test-set big \
  --batch-size 8 \
  --save-dir ./results
```

### Example 3: Model Comparison

```bash
uv run python models/main.py \
  --compare-models "gpt 4o-mini" "gpt 5-mini" "gemini 2.5-flash" \
  --test-set simple \
  --batch-size 4
```

### Example 4: Batch Size Optimization

```bash
uv run python models/main.py \
  --engine gpt \
  --model 4o-mini \
  --test-set simple \
  --compare 1 2 4 8 16
```

### Example 5: Running with Different Logging Levels

```bash
# Detailed logging (all batch processing and metrics)
uv run python models/main.py \
  --engine gemini \
  --model 2.5-flash \
  --test-set simple \
  --debugging detailed

# Output only (just final results and progress bars, default)
uv run python models/main.py \
  --engine gpt \
  --model 4o-mini \
  --test-set simple \
  --debugging output

# Silent mode (no output)
uv run python models/main.py \
  --engine gpt \
  --model 4o-mini \
  --test-set simple \
  --debugging none
```

## Project Structure

```
quran/
├── models/           # Model implementations (GPT, Gemini, open-source)
├── helpers/          # Helper utilities
├── config/           # Configuration files
├── data/             # Data files and test sets
├── pyproject.toml    # Project dependencies
└── README.md         # This file
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
