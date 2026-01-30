# Novel Job Matching

This is the coding part of Bachelor thesis by Seyed Taha Amirhosseini

## Directories

- `conception/` : contains the files from exploration and conception phase.
- `implementation/` : contains evaluation framework and job matching system. The dataset and results of the evaluation of LLMs can be found there.
- `evaluation/` : contains code from evaluation phase. code for all the figures used in the thesis can be found in this directory. It also contains the LinkedIn comparison code and results.

## Setup

This project uses `uv`.

```powershell
uv sync
```

## Run (Job Matching System API)

From `implementation/job_matching_system/`:

```powershell
uv run uvicorn app:app
```
