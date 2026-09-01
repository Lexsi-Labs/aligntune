# Raw Factory CLI Configurations

`aligntune train --config` accepts backend-factory argument names directly.
The `type` field selects the high-level trainer. RL also requires a concrete
`algorithm`.

Run a configuration with:

```bash
aligntune train --config path/to/config.yaml
```

## SFT

```yaml
type: sft
model_name: gpt2
dataset_name: tatsu-lab/alpaca
backend: trl
output_dir: ./outputs/cli_sft
num_epochs: 1
batch_size: 4
learning_rate: 0.00002
max_seq_length: 512
column_mapping:
  instruction: instruction
  input: input
  output: output
```

```bash
aligntune train \
  --type sft \
  --model gpt2 \
  --dataset tatsu-lab/alpaca \
  --backend trl \
  --batch-size 4 \
  --epochs 1
```

## RL

```yaml
type: rl
algorithm: grpo
model_name: Qwen/Qwen2.5-0.5B-Instruct
dataset_name: openai/gsm8k
backend: trl
output_dir: ./outputs/cli_grpo
num_epochs: 1
batch_size: 4
num_generations: 4
max_seq_length: 512
max_completion_length: 128
learning_rate: 0.000005
```

```bash
aligntune train \
  --type rl \
  --algorithm grpo \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset openai/gsm8k \
  --backend trl \
  --batch-size 4 \
  --num-generations 4
```

RL algorithm shorthand also works:

```bash
aligntune train --type dpo --model gpt2 --dataset Anthropic/hh-rlhf
```

## Distillation

```yaml
type: distill
student_model: Qwen/Qwen2.5-0.5B
teacher_model: Qwen/Qwen2.5-1.5B
dataset_name: tatsu-lab/alpaca
backend: trl
output_dir: ./outputs/cli_distill
batch_size: 2
num_epochs: 1
temperature: 3.0
alpha: 0.5
max_seq_length: 512
```

```bash
aligntune train \
  --type distill \
  --student-model Qwen/Qwen2.5-0.5B \
  --teacher-model Qwen/Qwen2.5-1.5B \
  --dataset tatsu-lab/alpaca \
  --backend trl
```

## Evolution Strategies

```yaml
type: es
model_name: gpt2
dataset_name: openai/gsm8k
backend: es
output_dir: ./outputs/cli_es
population_size: 8
sigma: 0.1
num_iterations: 10
learning_rate: 0.01
max_seq_length: 512
```

```bash
aligntune train \
  --type es \
  --model gpt2 \
  --dataset openai/gsm8k \
  --population-size 8 \
  --num-iterations 10
```

## Tokenization

```yaml
type: tokenization
base_model: gpt2
target_languages:
  - ar
dataset_name: wikimedia/wikipedia
config_name: 20231101.ar
split: train
output_dir: ./outputs/cli_tokenization
num_new_tokens: 1000
extension_method: continued_bpe
```

```bash
aligntune train \
  --type tokenization \
  --base-model gpt2 \
  --target-language ar \
  --dataset wikimedia/wikipedia \
  --num-new-tokens 1000 \
  --extension-method continued_bpe
```

All keys other than `type`, `training_type`, and `algo` are forwarded to the
selected factory unchanged. Those three keys are dispatch controls and are
not sent as trainer arguments.
