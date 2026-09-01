"""
Comprehensive Testing Script for AlignTune RL Configuration Handling

Tests various config scenarios:
1. Direct parameter passing
2. YAML config loading
3. Dict config loading
4. Mixed config + parameter override
5. Different algorithms (DPO, PPO, GRPO, GSPO, etc.)
6. Backend selection (TRL, Unsloth)
7. Reward model configurations
8. Advanced features (counterfactual GRPO, GBMPO, etc.)
"""

import os
import sys
import tempfile
import yaml
import traceback
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from aligntune.core.backend_factory import create_rl_trainer, list_backends


class TestRLConfigHandling:
    """Test suite for RL configuration handling"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = tempfile.mkdtemp()
        
    def log_test(self, test_name: str, status: str, message: str = ""):
        """Log test result"""
        self.test_results[test_name] = {
            "status": status,
            "message": message
        }
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if message:
            print(f"   → {message}")
    
    def create_yaml_config(self, config_dict: Dict[str, Any]) -> str:
        """Create temporary YAML config file"""
        config_path = os.path.join(self.temp_dir, f"test_config_{len(self.test_results)}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
        return config_path
    
    # ==================== TEST 1: Direct Parameters ====================
    def test_direct_parameters(self):
        """Test 1: Create trainer with direct parameters only"""
        test_name = "Test 1: Direct Parameters"
        try:
            trainer = create_rl_trainer(
                model_name="Qwen/Qwen2.5-0.5B",
                dataset_name="google-research-datasets/mbpp",
                algorithm='grpo',
                column_mapping={
                    "text": "prompt",
                    "code": "response"
                },
                split="train[:1%]",
                backend="trl",
                output_dir=os.path.join(self.temp_dir, "output_direct"),
                batch_size=8,
                num_generations=4,
                rewards=[{
                    "type": "mbpp_reward",
                    "weight": 1.0,
                    "params": {}
                }],
                learning_rate=2e-4,
                max_seq_length=512,
            )
            
            # Verify config was created correctly
            assert trainer.config.model.name_or_path == "Qwen/Qwen2.5-0.5B", \
                f"Expected model 'Qwen/Qwen2.5-0.5B', got '{trainer.config.model.name_or_path}'"
            assert trainer.config.datasets[0].name == "google-research-datasets/mbpp", \
                f"Expected dataset 'google-research-datasets/mbpp', got '{trainer.config.datasets[0].name}'"
            assert trainer.config.algo == "grpo", \
                f"Expected algo 'grpo', got '{trainer.config.algo}'"
            assert trainer.config.train.per_device_batch_size == 8, \
                f"Expected batch_size 8, got {trainer.config.train.per_device_batch_size}"
            
            self.log_test(test_name, "PASS", "Direct parameters handled correctly")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}\n{traceback.format_exc()}")
    
    # ==================== TEST 2: YAML Config ====================
    def test_yaml_config(self):
        """Test 2: Load config from YAML file"""
        test_name = "Test 2: YAML Config Loading"
        try:
            # Create YAML config with proper structure
            yaml_config = {
                "training_type": "rl",
                "algorithm": "dpo",
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-0.5B",
                    "max_seq_length": 512,
                },
                "datasets": [{
                    "name": "trl-lib/ultrafeedback_binarized",
                    "split": "train[:1%]",
                    "column_mapping": {
                        "prompt": "prompt",
                        "chosen": "chosen",
                        "rejected": "rejected"
                    }
                }],
                "train": {
                    "per_device_batch_size": 4,
                    "learning_rate": 1e-4,
                    "beta": 0.1,
                    "loss_type": "sigmoid",
                    "output_dir": os.path.join(self.temp_dir, "output_yaml"),
                },
                "backend": "trl",
            }
            
            config_path = self.create_yaml_config(yaml_config)
            
            trainer = create_rl_trainer(config=config_path)
            
            # Verify YAML values were loaded
            assert trainer.config.model.name_or_path == "Qwen/Qwen2.5-0.5B", \
                f"Model name mismatch: {trainer.config.model.name_or_path}"
            assert trainer.config.algo == "dpo", \
                f"Algorithm mismatch: {trainer.config.algo}"
            assert trainer.config.train.per_device_batch_size == 4, \
                f"Batch size mismatch: {trainer.config.train.per_device_batch_size}"
            assert trainer.config.train.beta == 0.1, \
                f"Beta mismatch: {trainer.config.train.beta}"
            
            self.log_test(test_name, "PASS", f"YAML config loaded from {os.path.basename(config_path)}")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}")
    
    # ==================== TEST 3: Dict Config ====================
    def test_dict_config(self):
        """Test 3: Load config from dictionary"""
        test_name = "Test 3: Dict Config Loading"
        try:
            dict_config = {
                "training_type": "rl",
                "algorithm": "ppo",
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-0.5B",
                    "reward_model_name": "Qwen/Qwen2.5-0.5B",
                },
                "datasets": [{
                    "name": "trl-lib/ultrafeedback_binarized",
                    "split": "train[:1%]",
                }],
                "train": {
                    "per_device_batch_size": 2,
                    "num_ppo_epochs": 4,
                    "kl_coef": 0.05,
                    "output_dir": os.path.join(self.temp_dir, "output_dict"),
                },
                "backend": "trl",
            }
            
            trainer = create_rl_trainer(config=dict_config)
            
            # Verify dict values were loaded
            assert trainer.config.algo == "ppo", \
                f"Algorithm mismatch: {trainer.config.algo}"
            assert trainer.config.train.num_ppo_epochs == 4, \
                f"PPO epochs mismatch: {trainer.config.train.num_ppo_epochs}"
            assert trainer.config.train.kl_coef == 0.05, \
                f"KL coef mismatch: {trainer.config.train.kl_coef}"
            
            self.log_test(test_name, "PASS", "Dict config handled correctly")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}")
    
    # ==================== TEST 4: Mixed Config + Override ====================
    def test_config_override(self):
        """Test 4: YAML config with parameter override"""
        test_name = "Test 4: Config Override"
        try:
            # Create base YAML config with proper structure
            yaml_config = {
                "training_type": "rl",
                "algorithm": "grpo",
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-0.5B",
                },
                "datasets": [{
                    "name": "google-research-datasets/mbpp",
                    "split": "train[:1%]",
                }],
                "train": {
                    "per_device_batch_size": 4,
                    "learning_rate": 2e-4,
                },
                "backend": "trl",
            }
            
            config_path = self.create_yaml_config(yaml_config)
            
            # Override with direct parameters
            trainer = create_rl_trainer(
                config=config_path,
                batch_size=8,  # Override YAML value
                num_generations=16,  # New parameter
                output_dir=os.path.join(self.temp_dir, "output_override"),
                temperature=0.7,  # New parameter
            )
            
            # Verify override worked
            assert trainer.config.train.per_device_batch_size == 8, \
                f"Batch size override failed: {trainer.config.train.per_device_batch_size} != 8"
            assert trainer.config.train.num_generations == 16, \
                f"Num generations failed: {trainer.config.train.num_generations} != 16"
            assert trainer.config.train.temperature == 0.7, \
                f"Temperature failed: {trainer.config.train.temperature} != 0.7"
            assert trainer.config.train.learning_rate == 2e-4, \
                f"Learning rate from YAML failed: {trainer.config.train.learning_rate} != 2e-4"
            
            self.log_test(test_name, "PASS", "Config override successful")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}")
    
    # ==================== TEST 5: GSPO Algorithm ====================
    def test_gspo_algorithm(self):
        """Test 5: GSPO algorithm configuration"""
        test_name = "Test 5: GSPO Algorithm"
        try:
            trainer = create_rl_trainer(
                model_name="Qwen/Qwen2.5-0.5B",
                dataset_name="google-research-datasets/mbpp",
                algorithm='gspo',
                backend="trl",  # GSPO only supported by TRL
                split="train[:1%]",
                output_dir=os.path.join(self.temp_dir, "output_gspo"),
                batch_size=4,
                gspo_gamma=0.1,
                gspo_delta=0.1,
                rewards=[{
                    "type": "mbpp_reward",
                    "weight": 1.0,
                    "params": {}
                }],
            )
            
            assert trainer.config.algo == "gspo", \
                f"Algorithm mismatch: {trainer.config.algo} != gspo"
            assert trainer.config.train.gspo_gamma == 0.1, \
                f"GSPO gamma mismatch: {trainer.config.train.gspo_gamma} != 0.1"
            
            self.log_test(test_name, "PASS", "GSPO algorithm configured")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}")
    
    # ==================== TEST 6: Counterfactual GRPO ====================
    def test_counterfactual_grpo(self):
        """Test 6: Counterfactual GRPO with specific parameters"""
        test_name = "Test 6: Counterfactual GRPO"
        try:
            trainer = create_rl_trainer(
                model_name="Qwen/Qwen2.5-0.5B",
                dataset_name="google-research-datasets/mbpp",
                algorithm='counterfact_grpo',
                backend="trl",
                split="train[:1%]",
                output_dir=os.path.join(self.temp_dir, "output_counterfact"),
                batch_size=4,
                # Counterfactual-specific params
                boost_factor=2.0,
                min_weight=0.5,
                max_spans=10,
                answer_weight=1.5,
                method_name="counterfactual",
                enable_gradient_conservation=True,
                rewards=[{
                    "type": "mbpp_reward",
                    "weight": 1.0,
                    "params": {}
                }],
            )
            
            assert trainer.config.algo == "counterfact_grpo", \
                f"Algorithm mismatch: {trainer.config.algo} != counterfact_grpo"
            assert trainer.config.train.boost_factor == 2.0, \
                f"Boost factor mismatch: {trainer.config.train.boost_factor} != 2.0"
            assert trainer.config.train.max_spans == 10, \
                f"Max spans mismatch: {trainer.config.train.max_spans} != 10"
            
            self.log_test(test_name, "PASS", "Counterfactual GRPO configured")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}")
    
    # ==================== TEST 7: GBMPO Variants ====================
    def test_gbmpo_variants(self):
        """Test 7: GBMPO algorithm variants"""
        test_name = "Test 7: GBMPO Variants"
        try:
            # Test GBMPO with L2KL divergence
            trainer = create_rl_trainer(
                model_name="Qwen/Qwen2.5-0.5B",
                dataset_name="google-research-datasets/mbpp",
                algorithm='gbmpo',
                backend="trl",
                split="train[:1%]",
                output_dir=os.path.join(self.temp_dir, "output_gbmpo"),
                batch_size=4,
                gbmpo_divergence_type="l2kl",
                gbmpo_l2_coefficient=0.0001,
                gbmpo_epsilon=0.2,
                rewards=[{
                    "type": "mbpp_reward",
                    "weight": 1.0,
                    "params": {}
                }],
            )
            
            assert trainer.config.algo == "gbmpo", \
                f"Algorithm mismatch: {trainer.config.algo} != gbmpo"
            assert trainer.config.train.gbmpo_divergence_type == "l2kl", \
                f"Divergence type mismatch: {trainer.config.train.gbmpo_divergence_type} != l2kl"
            assert trainer.config.train.gbmpo_l2_coefficient == 0.0001, \
                f"L2 coef mismatch: {trainer.config.train.gbmpo_l2_coefficient} != 0.0001"
            
            self.log_test(test_name, "PASS", "GBMPO variant configured")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}")
    
    # ==================== TEST 8: Reward Model Configurations ====================
    def test_reward_models(self):
        """Test 8: Different reward model configurations"""
        test_name = "Test 8: Reward Model Configs"
        try:
            # Test with pretrained reward model
            trainer = create_rl_trainer(
                model_name="Qwen/Qwen2.5-0.5B",
                dataset_name="trl-lib/ultrafeedback_binarized",
                algorithm='dpo',
                backend="trl",
                split="train[:1%]",
                output_dir=os.path.join(self.temp_dir, "output_reward"),
                batch_size=4,
                reward_model_name="Qwen/Qwen2.5-0.5B",
                reward_device="auto",
            )
            
            assert trainer.config.model.reward_model_name == "Qwen/Qwen2.5-0.5B", \
                f"Reward model mismatch: {trainer.config.model.reward_model_name}"
            assert trainer.config.model.reward_device == "auto", \
                f"Reward device mismatch: {trainer.config.model.reward_device}"
            
            self.log_test(test_name, "PASS", "Reward model configured")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}")
    
    # ==================== TEST 9: Multiple Reward Functions ====================
    def test_multiple_rewards(self):
        """Test 9: Multiple reward functions"""
        test_name = "Test 9: Multiple Rewards"
        try:
            trainer = create_rl_trainer(
                model_name="Qwen/Qwen2.5-0.5B",
                dataset_name="google-research-datasets/mbpp",
                algorithm='grpo',
                backend="trl",
                split="train[:1%]",
                output_dir=os.path.join(self.temp_dir, "output_multi_reward"),
                batch_size=4,
                rewards=[
                    {
                        "type": "mbpp_reward",
                        "weight": 1.0,
                        "params": {}
                    },
                    {
                        "type": "length_penalty",
                        "weight": 0.5,
                        "params": {"target_length": 100}
                    }
                ],
            )
            
            assert len(trainer.config.rewards) == 2, \
                f"Rewards count mismatch: {len(trainer.config.rewards)} != 2"
            assert trainer.config.rewards[0]["type"] == "mbpp_reward", \
                f"First reward type mismatch: {trainer.config.rewards[0]['type']}"
            assert trainer.config.rewards[1]["type"] == "length_penalty", \
                f"Second reward type mismatch: {trainer.config.rewards[1]['type']}"
            
            self.log_test(test_name, "PASS", "Multiple rewards configured")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}")
    
    # ==================== TEST 10: Backend Selection ====================
    def test_backend_selection(self):
        """Test 10: Backend selection (TRL vs Unsloth)"""
        test_name = "Test 10: Backend Selection"
        try:
            # Test TRL backend
            trainer_trl = create_rl_trainer(
                model_name="Qwen/Qwen2.5-0.5B",
                dataset_name="google-research-datasets/mbpp",
                algorithm='grpo',
                backend="trl",
                split="train[:1%]",
                output_dir=os.path.join(self.temp_dir, "output_trl"),
                batch_size=4,
            )
            
            # Verify environment is set for TRL
            assert os.environ.get('PURE_TRL_MODE') == '1', \
                f"PURE_TRL_MODE not set correctly: {os.environ.get('PURE_TRL_MODE')}"
            
            self.log_test(test_name, "PASS", "Backend selection working")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}")
    
    # ==================== TEST 11: Complex YAML Config ====================
    def test_complex_yaml_config(self):
        """Test 11: Complex YAML with nested configs"""
        test_name = "Test 11: Complex YAML Config"
        try:
            yaml_config = {
                "training_type": "rl",
                "algorithm": "grpo",
                "model": {
                    "name_or_path": "Qwen/Qwen2.5-0.5B",
                    "max_seq_length": 1024,
                },
                "datasets": [{
                    "name": "google-research-datasets/mbpp",
                    "split": "train[:1%]",
                    "column_mapping": {
                        "text": "prompt",
                        "code": "response"
                    }
                }],
                "train": {
                    "per_device_batch_size": 8,
                    "learning_rate": 2e-4,
                    "num_generations": 16,
                    "gradient_accumulation_steps": 4,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "save_steps": 100,
                    "logging_steps": 10,
                    "eval_steps": 50,
                    "output_dir": os.path.join(self.temp_dir, "output_complex"),
                },
                "rewards": [
                    {
                        "type": "mbpp_reward",
                        "weight": 1.0,
                        "params": {}
                    }
                ],
                "backend": "trl",
            }
            
            config_path = self.create_yaml_config(yaml_config)
            trainer = create_rl_trainer(config=config_path)
            
            # Verify complex config
            assert trainer.config.train.gradient_accumulation_steps == 4, \
                f"Gradient accumulation mismatch: {trainer.config.train.gradient_accumulation_steps} != 4"
            assert trainer.config.model.max_seq_length == 1024, \
                f"Max seq length mismatch: {trainer.config.model.max_seq_length} != 1024"
            assert trainer.config.train.temperature == 0.7, \
                f"Temperature mismatch: {trainer.config.train.temperature} != 0.7"
            assert trainer.config.train.save_steps == 100, \
                f"Save steps mismatch: {trainer.config.train.save_steps} != 100"
            
            self.log_test(test_name, "PASS", "Complex YAML config handled")
            
        except AssertionError as e:
            self.log_test(test_name, "FAIL", f"Assertion failed: {str(e)}")
        except Exception as e:
            self.log_test(test_name, "FAIL", f"Error: {str(e)}")
    
    # ==================== TEST 12: Error Handling ====================
    def test_error_handling(self):
        """Test 12: Error handling for invalid configs"""
        test_name = "Test 12: Error Handling"
        
        # Test 12a: Missing required field
        try:
            trainer = create_rl_trainer(
                # Missing model_name
                dataset_name="google-research-datasets/mbpp",
                algorithm='grpo',
            )
            self.log_test(test_name + " (Missing Field)", "FAIL", "Should have raised ValueError")
        except ValueError as e:
            self.log_test(test_name + " (Missing Field)", "PASS", f"Correctly raised: {str(e)}")
        except Exception as e:
            self.log_test(test_name + " (Missing Field)", "FAIL", f"Wrong exception: {type(e).__name__}: {str(e)}")
        
        # Test 12b: Invalid algorithm
        try:
            trainer = create_rl_trainer(
                model_name="Qwen/Qwen2.5-0.5B",
                dataset_name="google-research-datasets/mbpp",
                algorithm='invalid_algo',
            )
            self.log_test(test_name + " (Invalid Algo)", "FAIL", "Should have raised ValueError")
        except ValueError as e:
            self.log_test(test_name + " (Invalid Algo)", "PASS", f"Correctly raised: {str(e)}")
        except Exception as e:
            self.log_test(test_name + " (Invalid Algo)", "FAIL", f"Wrong exception: {type(e).__name__}: {str(e)}")
    
    # ==================== Run All Tests ====================
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("AlignTune RL Configuration Handling Test Suite")
        print("="*70 + "\n")
        
        # List available backends first
        print("Checking available backends...")
        list_backends()
        print()
        
        # Run tests
        self.test_direct_parameters()
        self.test_yaml_config()
        self.test_dict_config()
        self.test_config_override()
        self.test_gspo_algorithm()
        self.test_counterfactual_grpo()
        self.test_gbmpo_variants()
        self.test_reward_models()
        self.test_multiple_rewards()
        self.test_backend_selection()
        self.test_complex_yaml_config()
        self.test_error_handling()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)
        
        passed = sum(1 for r in self.test_results.values() if r["status"] == "PASS")
        failed = sum(1 for r in self.test_results.values() if r["status"] == "FAIL")
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        print("="*70 + "\n")
        
        if failed > 0:
            print("Failed Tests:")
            for test_name, result in self.test_results.items():
                if result["status"] == "FAIL":
                    print(f"  ❌ {test_name}")
                    print(f"     {result['message']}")
            print()


def main():
    """Main test runner"""
    tester = TestRLConfigHandling()
    tester.run_all_tests()


if __name__ == "__main__":
    main()