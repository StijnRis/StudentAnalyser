# StudentAnalyser

StudentAnalyser is a Python-based analytics toolkit for analysing fine-grained logs from classroom programming activities (edits, executions, chatbot interactions). The project contains data pipelines, analyzers and plotting utilities used in an honours study on student interactions with LLM-backed assistants.

## Quick start

Prerequisites

-   Python 3.10+ (use the interpreter in your environment)
-   Git

Create a virtual environment and install dependencies (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt
```

Run the main analysis (example):

```powershell
python main.py
```

## Project layout (top-level)

-   `chatbot.py` — helper scripts for chat-related processing
-   `main.py` — top-level entry point for running analyses
-   `pipeline/` — data pipeline scripts (e.g., `pipeline.py`, `jupyter_data_pipeline.py`)
-   `loader/` — data loading utilities (e.g., `load_excel_file.py`, `loader_pipeline.py`)
-   `executions/` — execution-related analysers (e.g., `execution_analyser.py`)
-   `edits/` — edit-event analysers (e.g., `edit_analyser.py`)
-   `interactions/` — chat interaction analysers (e.g., `interaction_analyser.py`)
-   `plots/` — plotting utilities
-   `writer/` — output writers (e.g., Excel export)

Explore the folders above to run specific analysers; each analyser script can typically be run directly (for example `python executions\execution_analyser.py`) but check the file header / docstring for exact usage.

## Data and privacy

This repository contains analysis code only. Raw student data used in the study is subject to consent and anonymization. Do not commit raw data or personal identifiers to this repository.

## JELAI and citation note

This work uses data collected with JELAI (Jupyter Environment for Learning Analytics and AI) and a vscode tutor extension. If you refer to JELAI in writing, please cite:

Valle Torre et al., "JELAI: Integrating AI and Learning Analytics in Jupyter Notebooks" (2025)

## License & contact

Check the repository root for a `LICENSE` file. For questions, open an issue or contact the repository owner.
