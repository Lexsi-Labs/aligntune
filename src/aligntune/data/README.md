# Unified Dataset Handler for AlignTune

The **Unified Dataset Handler** (`DataManager`) is the central engine for loading, normalizing, and preparing data for all AlignTune algorithms (SFT, DPO, GRPO, etc.). It eliminates the need for manual data parsing and ensures consistency across training and evaluation.


## 🚀 Quick Start

Dataset can be loaded simply with a **Task** and **Source**.

```{python}
from aligntune.data.manager import DataManager

# 1. Initialize for your specific task (e.g., SFT, DPO, GRPO)
manager = DataManager(task_type="sft", system_prompt="You are a helpful assistant.")

# 2. Load from HuggingFace, Local File, or Folder
dataset = manager.load_dataset("tatsu-lab/alpaca")

# 3. Use the standardized splits
train_data = dataset["train"]
eval_data = dataset["validation"]
```


## 🎯 Supported Tasks & Schemas

The manager automatically detects column names based on the task type. It standardizes them into the internal format required by AlignTune trainers.

| Task Type | Config `task_type` | Input Columns (Auto-Detected) | Standardized Output Columns |
| :--- | :---: | :--- | :--- | 
| *SFT* | `"sft"` | `instruction`, `input`, `content`, `dialogue`... | `prompt`, `completion` |
| *DPO* | `"dpo"` | `winner`, `loser`, `better`, `worse`... | `prompt`, `chosen`, `rejected` |
| *GRPO* | `"grpo"` | `question`, `query`, `problem`... | `prompt`, `response` |
| *QA* | `"qa"` | `context`, `passage`, `article`... | `context`, `question`, `answer` |
| *Summary* | `"summarization"` | `article`, `text`, `content`... | `document`, `summary` |


## 📂 Supported Data Sources

* HuggingFace Hub: `manager.load_dataset("openai/gsm8k", config_name="main")`

* JSON / JSONL: `manager.load_dataset("./data/train.jsonl")`

* CSV: `manager.load_dataset("./data/dataset.csv", delimiter=",")`

* Parquet: `manager.load_dataset("./data/large_dataset.parquet")`

* Local Directory: `manager.load_dataset("./my_data_folder/") (Loads all compatible files inside)`


## ✨ Key Features

### 1. Automatic Column Mapping

You don't need to rename columns manually. The system uses heuristics to find the right data.

* Input: `{"instruction": "Hi", "output": "Hello"}`
* Output (SFT): `{"prompt": "Hi", "completion": "Hello"}`

### 2. Smart Splitting

Ensures you always have `train`, `validation`, and `test` splits for reliable evaluation.

* **1 Split Found:** Splits into **80% Train / 10% Val / 10% Test**.
* **2 Splits Found:** Uses larger as **Train**, smaller as **Test**. Creates **Validation** from Train (90/10).
* **3+ Splits Found:** Uses standard mapping (Train/Val/Test) based on size.

### 3. System Prompt Injection

Seamlessly injects system prompts into both plain text and chat formats.

* Text: Prepends to prompt: `System Prompt\n\nUser Input`
* Chat: Inserts as first message: `[{"role": "system", "content": "..."}, ...]`

## 🛠️ Advanced Usage

### Custom Column Mapping

If your dataset uses unique column names that auto-detection misses, you can map them manually.

```{python}
manager = DataManager(
    task_type="sft",
    column_mapping={
        "my_weird_input": "prompt",
        "human_label": "completion"
    }
)
```


### Custom Processing Functions

Apply arbitrary Python functions (cleaning, filtering, formatting) during the loading pipeline.

```{python}
def clean_text(example):
    example["prompt"] = example["prompt"].strip().lower()
    return example

manager = DataManager(
    task_type="sft",
    processing_fn=clean_text,
    processing_batched=False
)
```

## ❓ Troubleshooting

**Error**: `Dataset missing required columns ['prompt']`

* **Cause**: Your dataset columns didn't match any known synonyms in the TaskSchema.
* **Fix**: Use the `column_mapping` argument to explicitly tell the manager which column is which.

**Error**: `Unknown split "train"`

* Cause: The dataset (e.g., UltraFeedback) uses non-standard split names like `train_prefs`.
* Fix: Specify the split explicitly: `manager.load_dataset("repo/id", split="train_prefs")`.

Warning: `Dataset too small to split`

* Cause: Your dataset has fewer than ~5 examples.
* Fix: Provide more data! The splitter needs enough samples to create meaningful validation sets.
