from aligntune.core.backend_factory import create_rl_trainer


algo = ['grpo','dpo','dapo','gspo','dapo','drgrpo','counterfact_grpo','gbmpo','nmgrpo','bolt','ppo']
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
                    "prompt": example["prompt"],
                    "chosen": example["answer"],
                    "rejected": " None , 0 "
                }
        trainer = create_rl_trainer(
                model_name="Qwen/Qwen3-0.6B",
                dataset_name="openai/gsm8k",
                config_name='main',
                algorithm=a,
                backend="trl",
                output_dir="./output/grpo_trl",
                batch_size=8,
                processing_fn=preprocess_function,
            )
    else:
        trainer = create_rl_trainer(
                model_name="Qwen/Qwen3-0.6B",
                dataset_name="openai/gsm8k",
                config_name='main',
                algorithm=a,
                backend="trl",
                output_dir="./output/grpo_trl",
                batch_size=8,
            )
    trainer.setup_model()
    trainer.setup_data()
    trainer.setup_rewards()
    trainer.setup_trainer()
    try:
        eval_data_10 = trainer.eval_dataset.select(range(2))
        trainer.evaluate(eval_dataset=eval_data_10,use_custom_evaluator=False)
        trainer.evaluate(eval_dataset=eval_data_10,use_custom_evaluator=True)
        output['pass']+=1

        pass_algos.append(a)
    except Exception as e:
        print(e)
        output['fail']+=1
        failed_algos.append(a)
print(f"Passed tests: {output['pass']}")
print(f"Failed tests: {output['fail']} {failed_algos}")



