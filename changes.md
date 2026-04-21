# Changes Log

This file tracks changes made with AI assistance so there is a transparent record of tool usage and contributions.

## How to use this log

- I will append a short entry here each time I make a code or project-file change.
- You can use this as supporting material when writing your AI disclosure section.
- Suggested interpretation:
  - "AI contribution" = what I changed or drafted
  - "Your contribution" = what you decided, reviewed, edited, tested, or explained in your own words

## Entries

### 2026-04-18

#### AI change

- Created `changes.md` in the repository root.
- Added a simple structure for logging future AI-assisted edits.

#### User contribution

- Requested a persistent markdown log to support transparent AI-use disclosure.

### 2026-04-19

#### AI change

- Explained how overlapping next-token windows can cause train/validation leakage if the data is split after sequence construction.
- Recommended splitting the full token stream first and then building overlapping windows separately for training and validation.
- Flagged that `train_test_split(..., shuffle=False)` preserves order but still does not avoid overlap leakage if used after window creation.

#### User contribution

- Asked whether the current split approach maintained sequence order and whether it risked data leakage.
- Reviewed the explanation and chose to record this design correction for transparency.

### 2026-04-20

#### AI change

- Recommended adding `assert d_model % num_heads == 0` when defining multi-head attention so the hidden dimension can be split evenly across heads.
- Clarified that query, key, and value projections should typically map `d_model -> d_model` before reshaping into heads.
- Suggested the packed multi-head attention implementation that projects `Q`, `K`, and `V` with `Linear(d_model, d_model)` and then reshapes into `(num_heads, head_dim)` for efficiency.
- Suggested a top-level `TinyTransformerLM` class that ties together token embeddings, sinusoidal positional encoding, stacked `TransformerBlock`s, a final `RMSNorm`, and the output projection to vocabulary logits.
- Suggested an optional debugging comparison between the custom `SelfAttention` implementation and PyTorch's `nn.MultiheadAttention` reference by copying the custom `Q`, `K`, `V`, and output projection weights into the reference module, applying the same causal mask, and comparing output and attention-weight differences.
- Interpreted the resulting check as successful because the maximum output difference was `0.00000024` and the maximum attention-weight difference was `0.00000000`.

#### User contribution

- Asked whether multi-head attention should include a head count and requested that the design constraint be logged for AI-use transparency.
- Questioned the projection design, compared it against the per-head interpretation, and chose to adopt the packed implementation approach after discussion.
- Requested that the proposed top-level language model architecture be recorded in the AI-use log.
- Ran the debugging comparison in the notebook and reported the numerical differences for verification.

### 2026-04-21

#### AI change

- Reviewed the attention visualization cell against the assignment rubric and identified that the original heatmap code depended on undefined notebook-session variables (`attn`, `sample_idx`, `layer_idx`, and `head_idx`).
- Updated the notebook so the attention visualization cell runs the trained model under `torch.no_grad()` and extracts `attention_maps` directly before plotting.
- Expanded the visualization from a single heatmap to a 2x2 grid comparing two layers and two heads, making it easier to discuss differences across attention heads in the report.
- Corrected the 2x2 heatmap layout after review by moving the shared colorbar outside the subplot grid, reducing repeated axis-label clutter, and clearing the stale broken plot output from the notebook.
- Suggested report language focused on causal masking and local attention patterns, while avoiding unsupported claims about undertraining or head specialization.

#### User contribution

- Asked for a minimal fix to make the attention heatmap reproducible.
- Questioned whether written attention analysis belonged in the notebook or the PDF report and clarified the intended report-only analysis workflow.
- Rejected an overly speculative interpretation about diffuse attention and requested a more grounded explanation.
- Requested an improved visualization that could support a stronger attention-analysis score.
- Reviewed the generated 2x2 heatmap output, identified that the colorbar and labels overlapped the plots, and requested a correction.

#### AI change

- Installed `nbconvert` into the project virtual environment and used it to generate `A3.py` from `A3.ipynb`.
- Removed the notebook-only `get_ipython().system(...)` install line from `A3.py` so the script can run with plain Python.
- Updated the generated script to reuse `data/tinyshakespeare.txt` when present instead of redownloading it every run.
- Added `requirements.txt` with the project dependencies, including `nbconvert`.
- Replaced the placeholder `README.md` with setup, run, file overview, model details, and reproducibility notes.

#### User contribution

- Requested a script version of the notebook, dependency file, and README for repository submission.

#### AI change

- Added a project `.gitignore` for Python, Jupyter, virtual environment, cache, editor, and temporary files.

#### User contribution

- Requested a proper `.gitignore` so local environment files are not committed.

#### AI change

- Updated `README.md` so the notebook-running instructions appear before the Python script instructions.

#### User contribution

- Requested that the README prioritize running the notebook as the main workflow.

#### AI change

- Clarified in `README.md` that environment setup is only needed if the notebook environment does not already have the required dependencies, and that the same environment works for both notebook and script execution.

#### User contribution

- Requested a clearer setup note for users running the notebook.

#### AI change

- Removed the notebook dependency-install cell now that dependencies are documented in `requirements.txt` and `README.md`.
- Regenerated `A3.py` from the cleaned notebook and kept the script from redownloading the dataset when it already exists.

#### User contribution

- Pointed out that the notebook no longer needs a `!pip install ...` cell after adding setup instructions.

#### AI change

- Updated `A3.py` so the script creates an `outputs/` directory and saves `training_validation_loss.png` and `attention_heatmaps.png`.
- Updated `README.md` to document the generated figure output paths.

#### User contribution

- Requested that the Python script write graph and heatmap outputs to a directory.
