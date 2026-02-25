# Recipe Testing Roadmap

This document tracks the testing status of all AlignTune recipes.

## Testing Categories

### ✅ Configuration Validation
**Status**: COMPLETE (17/17)

All recipes have been validated for:
- [x] YAML structure correctness
- [x] Required fields present
- [x] Parameter types correct
- [x] No syntax errors

### ⏳ End-to-End Training Tests
**Status**: PENDING (0/17)

Full training runs from start to finish are pending for all recipes.

---

## Priority Testing Queue

### Phase 1: Core Functionality (Priority: HIGH)
These recipes cover the most common use cases and should be tested first.

| Recipe | Algorithm | Model Size | Est. Time | Est. VRAM | Status |
|--------|-----------|------------|-----------|-----------|--------|
| `qwen-math-grpo` | GRPO | 3B | 2-3h | 18GB | ⏳ Pending |
| `llama3-ultrafeedback-dpo` | DPO | 8B | 2-3h | 24GB | ⏳ Pending |
| `qwen-instruct-ultrachat` | SFT | 3B | 2-3h | 16GB | ⏳ Pending |
| `gemma-instruct-finetome` | SFT | 2B | 1-2h | 12GB | ⏳ Pending |
| `qwen-mbpp-bolt` | BOLT | 1.7B | 2-3h | 16GB | ⏳ Pending |

**Why these first?**
- Most commonly requested algorithms (SFT, DPO, GRPO, BOLT)
- Relatively fast to train (1-3 hours)
- Cover key use cases (instruction, math, code, alignment)
- Efficient models (good for testing infrastructure)

---

### Phase 2: Extended Coverage (Priority: MEDIUM)
Standard algorithms with larger models or longer training times.

| Recipe | Algorithm | Model Size | Est. Time | Est. VRAM | Status |
|--------|-----------|------------|-----------|-----------|--------|
| `llama3-instruct-orca` | SFT | 8B | 3-4h | 24GB | ⏳ Pending |
| `llama3-gsm8k-grpo` | GRPO | 8B | 3-4h | 28GB | ⏳ Pending |
| `qwen-preferences-dpo` | DPO | 3B | 1-2h | 16GB | ⏳ Pending |
| `mistral-hhrlhf-dpo` | DPO | 7B | 3-4h | 28GB | ⏳ Pending |
| `gemma-helpfulness-dpo` | DPO | 2B | 1-2h | 12GB | ⏳ Pending |
| `deepseek-humaneval-grpo` | GRPO | 1.3B | 1-2h | 12GB | ⏳ Pending |
| `llama-code-bolt` | BOLT | 3B | 2-3h | 18GB | ⏳ Pending |
| `llama3-helpful-harmless-ppo` | PPO | 8B | 4-6h | 32GB | ⏳ Pending |
| `qwen-safety-ppo` | PPO | 3B | 2-3h | 20GB | ⏳ Pending |

**Testing Goals**:
- Validate configurations across all standard algorithms
- Test across different model sizes (1B - 8B)
- Verify VRAM estimates
- Confirm training time estimates

---

### Phase 3: Advanced Algorithms (Priority: LOW)
Research-grade algorithms with longer training times.

| Recipe | Algorithm | Model Size | Est. Time | Est. VRAM | Status |
|--------|-----------|------------|-----------|-----------|--------|
| `neural-mirror-grpo-math` | NMGRPO | 7B | 8-12h | 32GB | ⏳ Pending |
| `dr-grpo-robust-math` | DR-GRPO | 8B | 4-6h | 28GB | ⏳ Pending |
| `gbmpo-l2-code` | GBMPO | 6.7B | 3-4h | 24GB | ⏳ Pending |

**Testing Goals**:
- Validate advanced algorithm implementations
- Benchmark against standard GRPO
- Document performance improvements
- Verify convergence behavior

---

## Testing Protocol

For each recipe, perform the following:

### 1. Pre-Test Checklist
- [ ] GPU available with sufficient VRAM
- [ ] Dataset accessible (authentication if needed)
- [ ] Disk space available for checkpoints
- [ ] Monitoring tools ready (wandb/tensorboard)

### 2. Training Run
```bash
# Start training
aligntune recipes run <recipe-name>

# Or with custom config
aligntune recipes copy <recipe-name> --output test_config.yaml
# Edit test_config.yaml as needed
aligntune train --config test_config.yaml
```

### 3. Metrics to Collect
- **Hardware**: GPU model, VRAM used, actual memory peak
- **Time**: Actual training time (vs estimated)
- **Convergence**: Final loss, training curve shape
- **Quality**: Evaluation metrics (if applicable)
- **Stability**: Any crashes, OOMs, or instabilities
- **Logs**: Save final training logs

### 4. Documentation
Update this file with:
- Actual vs estimated metrics
- Any configuration adjustments needed
- Recommendations for users
- Known issues or limitations

---

## Test Results Template

```markdown
## Recipe: <recipe-name>

**Tested By**: <your-name>  
**Date**: YYYY-MM-DD  
**Hardware**: <GPU model>  
**AlignTune Version**: <version>

### Configuration
- Modifications: <any changes from default>
- Reason: <why changes were needed>

### Results
- **Training Time**: Xh Ym (estimated: Ah Bm)
- **Peak VRAM**: XGB (estimated: YGB)
- **Final Loss**: X.XXXX
- **Convergence**: <smooth/unstable/etc>
- **Evaluation Metrics**: <if applicable>

### Issues Encountered
- <list any problems>

### Recommendations
- <suggested config changes>
- <user guidance>

### Status
- [x] Successfully completed
- [ ] Needs adjustment
- [ ] Failed (reason: ...)
```

---

## Community Contributions

We welcome community testing! Please:

1. Test any recipe from the list
2. Fill out the test results template
3. Submit via GitHub Issue or PR
4. Tag with `recipe-testing` label

**Benefits**:
- Help validate configurations
- Improve documentation
- Get acknowledged in CONTRIBUTORS
- Help the community

---

## Known Limitations

### Before Testing
- Recipes assume adequate VRAM (reduce batch size if OOM)
- Training times vary significantly by GPU generation
- Some datasets require HuggingFace authentication
- Advanced algorithms may need specific PyTorch versions

### Expected Adjustments
- **Batch sizes**: May need tuning per GPU
- **Learning rates**: May need adjustment per dataset
- **Generation length**: Depends on task requirements
- **Reward weights**: Task-specific optimization

---

## Contact

Questions about testing?
- GitHub Issues: Tag `recipe-testing`
- Email: support@lexsi.ai
- Documentation: See [recipes README](README.md)

---

**Last Updated**: January 2026  
**Next Review**: After Phase 1 completion
