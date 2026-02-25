# TAT-QA for Counterfactual GRPO: Dataset Report

## 1. What is TAT-QA?

**TAT-QA** (Tabular And Textual Question Answering) is a financial reasoning benchmark containing questions over real-world financial reports. Each example provides a financial table (income statements, balance sheets, cash flows, etc.) along with accompanying paragraph text, and asks a question that requires numerical reasoning to answer.

- **Source**: Real SEC filings and annual reports
- **Original paper**: Zhu et al., 2021 — *TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance*
- **License**: CC BY 4.0
- **HuggingFace**: `next-tat/TAT-QA`

### What we use

We filter to the **arithmetic subset only** — questions that require multi-step numerical computation (not simple span extraction or counting). This gives us:

| Split | Raw arithmetic | After filtering (<=1024 tokens) |
|-------|---------------|--------------------------------|
| Train | 5,552 | 5,009 |
| Test  | 706   | 630   |

We sample 2,000 training examples for our experiments. Median prompt length is **479 tokens** (max 1,023), meaning every example fits within `max_prompt_length: 1024`.

---

## 2. Data Format

Each preprocessed example has the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| `question` | Full context (paragraphs + table + question) | See examples below |
| `answer` | Clean numeric answer as a string | `"6.9"` |
| `derivation` | Human-annotated arithmetic expression | `(136,242 - 127,450)/127,450` |
| `scale` | Unit indicator: `""`, `"percent"`, `"thousand"`, `"million"` | `"percent"` |
| `answer_from` | Whether reasoning uses `"table"`, `"text"`, or `"table-text"` | `"table-text"` |

Only `question` and `answer` are used by the training pipeline. The other fields are preserved as metadata for analysis.

---

## 3. Full Examples

### Example A: Percentage Change

**Context + Question:**

```
GasLog Ltd. and its Subsidiaries
Notes to the consolidated financial statements (Continued)
For the years ended December 31, 2017, 2018 and 2019
(All amounts expressed in thousands of U.S. Dollars, except share and per share data)
14. Other Payables and Accruals
An analysis of other payables and accruals is as follows:
The unearned revenue represents charter hires received in advance in December 2019
relating to the hire period of January 2020 for 22 vessels (December 2018: 17 vessels).

|                   | As of December 31, |         |
| ----------------- | ------------------ | ------- |
|                   | 2018               | 2019    |
| Unearned revenue  | 38,680             | 48,183  |
| Accrued off-hire  | 7,376              | 6,968   |
| Accrued purchases | 18,578             | 9,759   |
| Accrued interest  | 38,107             | 36,746  |
| Other accruals    | 24,709             | 34,586  |
| Total             | 127,450            | 136,242 |

Question: What was the percentage change in total payables and accruals from 2018 to 2019?
```

**Gold answer**: `6.9` (percent)
**Derivation**: `(136,242 - 127,450) / 127,450`

**Expected model reasoning**:
> From the table, total payables were 127,450 in 2018 and 136,242 in 2019.
> The change is 136242 - 127450 = 8792.
> The percentage change is 8792 / 127450 = 0.069 = 6.9%.
> The answer is 6.9.

---

### Example B: Signed Difference with Negative Numbers

**Context + Question:**

```
A summary of additions and deductions related to the allowance for doubtful
accounts for the years ended March 31, 2019, 2018 and 2017 follows (amounts in millions):
(1) Deductions represent uncollectible accounts written off, net of recoveries.

|                                  | Balance at Beginning of Year | Additions | Deductions (1) | Balance at End of Year |
| -------------------------------- | ---------------------------- | --------- | -------------- | ---------------------- |
| Allowance for doubtful accounts: |                              |           |                |                        |
| Fiscal 2019                      | $2.2                         | $—        | $(0.2)         | $2.0                   |
| Fiscal 2018                      | $2.1                         | $0.2      | $(0.1)         | $2.2                   |
| Fiscal 2017                      | $2.5                         | $0.2      | $(0.6)         | $2.1                   |

Question: What was the percentage change in the deductions between 2018 and 2019?
```

**Gold answer**: `100` (percent)
**Derivation**: `(-0.2 - (-0.1)) / -0.1`

**Expected model reasoning**:
> From the table, deductions were $(0.1) in 2018 and $(0.2) in 2019.
> These are negative values: -0.1 and -0.2.
> The change is -0.2 - (-0.1) = -0.1.
> The percentage change is -0.1 / -0.1 = 1.0 = 100%.
> The answer is 100.

This example is particularly interesting because it requires handling parenthetical negative notation from financial statements and correctly computing percentage change with negative base values.

---

### Example C: Simple Difference with Scale

**Context + Question:**

```
REMUNERATION COMMITTEE REPORT
The table above shows the actual expenditure of the Group for employee pay
and distributions to shareholders compared to the retained earnings of the Group.

| RELATIVE IMPORTANCE OF SPEND ON PAY                 |       |       |       |
| --------------------------------------------------- | ----- | ----- | ----- |
| Expenditure USDm                                    | 2019  | 2018  | 2017  |
| Dividends paid                                      | -     | -     | 1.2   |
| Purchase of outstanding treasury shares in TORM A/S | -     | -     | -     |
| Purchase/disposals of treasury shares               | -     | -     | -     |
| Total                                               | -     | -     | 1.2   |
| Staff costs                                         | 45.8  | 46.2  | 43.8  |
| Retained earnings                                   | 920.0 | 752.0 | 786.0 |

Question: What was the change in retained earnings in 2019 from 2018?
```

**Gold answer**: `168` (million)
**Derivation**: `920.0 - 752.0`

**Expected model reasoning**:
> From the table, retained earnings were 752.0 in 2018 and 920.0 in 2019.
> The change is 920.0 - 752.0 = 168.0.
> The answer is 168.

---

## 4. Why Counterfactual GRPO is Well-Suited for This Dataset

### 4.1 The Core Mechanism

Counterfactual GRPO assigns **per-token importance weights** during training by asking: *"If I mask this part of the reasoning, does the model's answer probability drop?"*

For each model-generated completion:
1. **Detect arithmetic spans** — regex identifies patterns like `136242 - 127450 = 8792`
2. **Counterfactual masking** — replace each span with padding and re-run the forward pass
3. **Measure importance** — if masking a span causes the answer probability to drop, that span is *important*
4. **Upweight important tokens** — the training loss is scaled so critical reasoning steps get stronger gradient updates

### 4.2 Why Financial Table QA is a Strong Fit

**Multi-step arithmetic with clear intermediate calculations.**
Unlike classification tasks where the model outputs a single label, TAT-QA requires the model to:
1. **Extract** the right numbers from the table (e.g., `127,450` and `136,242`)
2. **Compute** intermediate results (e.g., `136242 - 127450 = 8792`)
3. **Derive** the final answer (e.g., `8792 / 127450 = 0.069`)

Each of these arithmetic steps is a detectable span that counterfactual masking can evaluate. The span detection regex `\d+(?:\.\d+)?(?:\s*[+\-×*/÷]\s*\d+(?:\.\d+)?)+\s*=\s*\d+(?:\.\d+)?` naturally catches model-generated calculations like `136242 - 127450 = 8792` and `8792 / 127450 = 0.069`.

**Not all reasoning steps are equally important.**
Consider Example A above. The model must:
- Read the table and identify the "Total" row (contextual understanding)
- Extract 127,450 and 136,242 (extraction — less critical, table is explicit)
- Compute 136242 - 127450 = 8792 (arithmetic — critical)
- Compute 8792 / 127450 = 0.069 (arithmetic — critical)
- Convert to percentage: 6.9% (formatting — moderately critical)

Vanilla GRPO treats all these tokens equally. Counterfactual GRPO will discover that the division step (8792 / 127450) is the most important — masking it completely destroys the answer probability — while the extraction step is less critical because the numbers are right there in the table. This **credit assignment** is exactly the advantage counterfactual brings.

**Structured format enables reliable span detection.**
The financial table + question format naturally elicits structured reasoning from the model:
```
Step 1: From the table, [value_1] in [year_1] is X and [value_2] in [year_2] is Y.
Step 2: The difference is X - Y = Z.
Step 3: The percentage change is Z / Y = W%.
```

This consistent structure means the arithmetic span detector will reliably find the computation steps, giving the counterfactual masking meaningful spans to evaluate. Unstructured reasoning (e.g., open-ended essay generation) would not provide such clear targets.

### 4.3 Expected Benefits Over Vanilla GRPO

| Aspect | Vanilla GRPO | Counterfactual GRPO |
|--------|-------------|-------------------|
| Token weighting | Uniform — all reasoning tokens get equal gradient | Adaptive — critical arithmetic steps get 2-4x gradient |
| Error patterns | May learn surface patterns (copy numbers from table) | Reinforces actual computation steps |
| Sample efficiency | Needs more examples to learn computation vs. extraction | Focuses learning on the hard part (arithmetic) from fewer examples |
| Robustness | Can overfit to formatting | Rewards process correctness, not just output format |

### 4.4 Comparison to GSM8K

| Property | GSM8K | TAT-QA (arithmetic) |
|----------|-------|---------------------|
| Domain | Grade-school word problems | Real financial reports |
| Context | Short natural language (50-100 tokens) | Table + paragraphs (400-1000 tokens) |
| Reasoning steps | 2-5 arithmetic operations | 1-3 operations (but harder extraction) |
| Answer format | `#### number` | Plain number |
| Challenge | Multi-step chaining | Extracting from structured tables + computing |

TAT-QA is complementary to GSM8K: it tests whether the counterfactual approach generalizes from simple word problems to **real-world structured data** where the difficulty is not just the arithmetic itself, but correctly identifying which numbers to use from a complex table.

---

## 5. Integration Summary

### Files Added/Modified

| File | Change |
|------|--------|
| `scripts/prepare_tatqa.py` | New — preprocessing script |
| `configs/qwen3_1.7b_tatqa_counterfact.yaml` | New — training config |
| `src/.../counterfact_grpo.py` line 243 | Added `'tatqa'` to math dataset detection |
| `src/.../counterfact_grpo.py` line 575 | Added `'tatqa'` to numeric grading path |

### Running

```bash
# Step 1: Preprocess (only needed once)
python scripts/prepare_tatqa.py --max_samples 2000 --max_tokens 1024

# Step 2: Train
python train_one.py --config configs/qwen3_1.7b_tatqa_counterfact.yaml
```

### Dataset Location

```
data/tatqa_arithmetic/
├── train/          # 2,000 examples (sampled, <=1024 tokens)
├── test/           # 630 examples (<=1024 tokens)
└── dataset_dict.json
```
