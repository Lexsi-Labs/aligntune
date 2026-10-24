# VHS tape for the SGLang rollout PR demo: https://github.com/charmbracelet/vhs
# Runs the unit tests and the live-server integration test on a GPU host over ssh.
# Precondition on that host: a checkout at ~/sglang-rl/aligntune with a venv at ~/sglang-rl/venv-at, and
#   python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 31234 \
#     --enable-lora --max-lora-rank 16 --lora-target-modules all --skip-tokenizer-init
# Render: DEMO_HOST=<ssh host> vhs examples/sglang_rollout_demo.tape

Output sglang_rollout_demo.gif
Require ssh
Require jq
Set Shell bash
Set FontSize 15
Set Width 1400
Set Height 720
Set TypingSpeed 15ms
# Model loading and torch imports dominate the wall time, so play back faster.
Set PlaybackSpeed 1.8

Hide
Type `export PS1='$ '; gpu() { ssh -q "$DEMO_HOST" "cd sglang-rl/aligntune && export PATH=\$PWD/../venv-at/bin:\$PATH OMP_NUM_THREADS=2 && $1"; }; clear`
Enter
Show

Type "# SGLang 0.5.10.post1 serves Qwen2.5-0.5B-Instruct with --enable-lora on the GPU host"
Enter
Type `gpu 'curl -s 127.0.0.1:31234/server_info' | jq '{version, model_path, enable_lora}'`
Enter
Wait+Screen@60s /"enable_lora": (true|false)/
Sleep 1500ms

Type "# Unit tests: fake SGLang server, no GPU"
Enter
Type "gpu 'pytest -qq -p no:cacheprovider tests/test_sglang_rollout.py'"
Enter
Wait+Screen@120s /\d+ passed|failed|rror/
Sleep 1500ms

Type "# Integration: random LoRA policy, adapter hot-load into SGLang, compare per-token logprobs"
Enter
Type "gpu 'SGLANG_URL=http://127.0.0.1:31234 pytest -qq -s -p no:cacheprovider tests/test_sglang_rollout_integration.py'"
Enter
Wait+Screen@180s /\b1 passed|\b1 failed|skipped|rror/
Sleep 4s
