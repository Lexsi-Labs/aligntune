from aligntune.core.backend_factory import create_rl_trainer

# Custom reward function
def my_math_reward(text, reference=None, **kwargs):
    """Simple reward: check if answer contains numbers"""
    print("This custom reard function is being used ")
    has_numbers = any(char.isdigit() for char in text)
    return 1.0 if has_numbers else 0.0

algos = ['grpo', 'dapo', 'gspo', 'drgrpo', 'counterfact_grpo', 'gbmpo', 'nmgrpo', 'bolt',]

print('='*80)
print('CUSTOM REWARD FUNCTION TEST FOR ALL ALGORITHMS')
print('='*80)

output = {
    'pass': 0,
    'fail': 0
}

pass_algos = []
failed_algos = []

for a in algos:
    print(f"\n{'='*60}")
    print(f"Testing algorithm: {a}")
    print(f"{'='*60}")
    
    try:
        # DPO needs special preprocessing
        if a == 'dpo':
            def preprocess_function(example):
                return {
                    "prompt": example["prompt"],
                    "chosen": example["answer"],
                    "rejected": "None, 0"
                }
            
            trainer = create_rl_trainer(
                model_name="Qwen/Qwen3-0.6B",
                dataset_name="openai/gsm8k",
                config_name='main',
                algorithm=a,
                backend="trl",
                output_dir=f"./output/{a}_test",
                batch_size=4,
                processing_fn=preprocess_function,
                rewards=[
                    {
                        "type": "custom",
                        "weight": 1.0,
                        "params": {
                            "reward_function": my_math_reward
                        }
                    }
                ]
            )
        else:
            trainer = create_rl_trainer(
                model_name="Qwen/Qwen3-0.6B",
                dataset_name="openai/gsm8k",
                config_name='main',
                algorithm=a,
                backend="trl",
                output_dir=f"./output/{a}_test",
                batch_size=4,
                rewards=[
                    {
                        "type": "custom",
                        "weight": 1.0,
                        "params": {
                            "reward_function": my_math_reward
                        }
                    }
                ]
            )
        
        # Setup rewards only
        print("  Setting up rewards...")
        trainer.setup_rewards()
        
        # Verify reward functions loaded
        print(f"  ✓ Loaded {len(trainer.reward_functions)} reward function(s)")
        
        # Test with sample completions
        test_completions = [
            "The answer is 42",
            "No numbers here"
        ]
        
        print("  Testing reward function...")
        rewards = trainer._combined_reward_function(test_completions)
        
        print(f"  Results: {rewards}")
        
        # Verify custom reward was called
        if rewards[0] > 0 and rewards[1] == 0:
            output['pass'] += 1
            pass_algos.append(a)
            print(f"  ✅ PASS: {a}")
        else:
            output['fail'] += 1
            failed_algos.append(a)
            print(f"  ❌ FAIL: {a} - Unexpected reward values")
            
    except Exception as e:
        print(f"  ❌ FAIL: {a}")
        print(f"  Error: {e}")
        output['fail'] += 1
        failed_algos.append(a)

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f"✅ Passed tests: {output['pass']}/{len(algos)}")
print(f"❌ Failed tests: {output['fail']}/{len(algos)}")

if pass_algos:
    print(f"\nPassed: {', '.join(pass_algos)}")
if failed_algos:
    print(f"\nFailed: {', '.join(failed_algos)}")
print('='*80)