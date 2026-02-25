from aligntune.core.backend_factory import create_rl_trainer


algo = ['grpo','dpo','dapo','gspo','dapo','drgrpo','counterfact_grpo','gbmpo','nmgrpo','bolt']
print('create_rl_trainer Test')
output = {
    'pass':0,
    'fail':0
    }

pass_algos = []
failed_algos = []
for a in algo:
    if a == 'dpo':
        def preprocess_function(example):
                return {
                    "prompt": example["question"],
                    "chosen": example["answer"],
                    "rejected": " None , 0 "
                }
        trainer = create_rl_trainer(
                model_name="Qwen/Qwen3-0.6B",
                dataset_name="openai/gsm8k",
                config_name='main',
                algorithm=a,
                backend="unsloth",
                output_dir="./output/grpo_trl",
                batch_size=8,
                # precison="fp16",
                processing_fn=preprocess_function,
            )
    else:
        trainer = create_rl_trainer(
                model_name="Qwen/Qwen3-0.6B",
                dataset_name="openai/gsm8k",
                config_name='main',
                algorithm=a,
                backend="unsloth",
                output_dir="./output/grpo_trl",
                batch_size=8,
                num_generations = 4,
                loss_type='grpo',
                # precision="fp16",
            )
   
    try:
        trainer.setup_model()
        trainer.setup_data()
        trainer.setup_rewards()
        trainer.setup_trainer()
        output['pass']+=1

        pass_algos.append(a)
    except Exception as e:
        print(e)
        output['fail']+=1
        failed_algos.append(a)
print(f"Passed tests: {output['pass']}")
print(f"Failed tests: {output['fail']} {failed_algos}")



