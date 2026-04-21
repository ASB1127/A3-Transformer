# A3-Transformer

Runnable code for Assignment 3: Transformer is All You Need.

This project implements a small Transformer language model for next-token prediction on the Tiny Shakespeare corpus. The model is intentionally lightweight and includes:

- Byte Pair Encoding tokenizer with vocabulary size 500
- Overlapping 50-token next-token prediction sequences
- 80/20 train/validation split
- Token embeddings plus sinusoidal positional encodings
- Custom causal multi-head self-attention
- Residual Transformer blocks with RMSNorm and feed-forward layers
- Cross-entropy training loss and validation perplexity
- Training/validation loss curve
- Attention heatmaps across layers and heads
- Debug comparison between the custom attention module and `torch.nn.MultiheadAttention`

## Files

- `A3.ipynb`: Original notebook version with embedded outputs and plots.
- `A3.py`: Runnable Python script version of the notebook.
- `requirements.txt`: Python dependencies needed to run the notebook or script.
- `data/tinyshakespeare.txt`: Tiny Shakespeare dataset.
- `changes.md`: AI-assistance change log for disclosure.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

To run the Python script:

```bash
python A3.py
```

The script uses `data/tinyshakespeare.txt` if it already exists. If the file is missing, it downloads the Tiny Shakespeare corpus from the public Karpathy `char-rnn` repository.

The script trains for 5 epochs with:

- `d_model = 128`
- `num_layers = 2`
- `num_heads = 4`
- `seq_length = 50`
- `batch_size = 64`
- `learning_rate = 1e-4`

After training, it prints validation perplexity and displays the loss curve and attention heatmaps.

## Reproducibility

The code sets Python, NumPy, and PyTorch random seeds to `42`. CUDA deterministic settings are enabled when a GPU is available.
