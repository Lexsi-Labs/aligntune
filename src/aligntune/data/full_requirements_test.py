import os
import json
import shutil
import pandas as pd
import logging
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from datasets import Dataset, load_dataset
from aligntune.data.manager import DataManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

TEST_DIR = "./temp_req_test"

def setup_env():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)

def teardown_env():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

def print_result(req_name: str, passed: bool, notes: str = ""):
    icon = "✅" if passed else "❌"
    print(f"{icon} {req_name.ljust(50)} : {notes}")

def main():
    print("================================================================")
    print("      FINETUNEHUB DATA REQUIREMENTS VERIFICATION MATRIX         ")
    print("================================================================\n")
    
    setup_env()

    # -------------------------------------------------------------------------
    # REQUIREMENT 1: Automatic column detection and mapping
    # -------------------------------------------------------------------------
    try:
        data = [{"instruction": "Q", "output": "A"}]
        with open(f"{TEST_DIR}/req1.json", "w") as f: json.dump(data, f)
        
        manager = DataManager(task_type="sft")
        ds = manager.load_dataset(f"{TEST_DIR}/req1.json")
        
        passed = "prompt" in ds["train"].column_names and "completion" in ds["train"].column_names
        print_result("Auto Column Detection & Mapping", passed, f"Mapped: {ds['train'].column_names}")
    except Exception as e:
        print_result("Auto Column Detection & Mapping", False, str(e))

    # -------------------------------------------------------------------------
    # REQUIREMENT 2: Support for multiple file formats (JSON, CSV, Parquet)
    # -------------------------------------------------------------------------
    try:
        # JSON
        with open(f"{TEST_DIR}/req2.json", "w") as f: json.dump([{"prompt": "j", "completion": "j"}], f)
        # CSV
        pd.DataFrame([{"prompt": "c", "completion": "c"}]).to_csv(f"{TEST_DIR}/req2.csv", index=False)
        # Parquet
        pd.DataFrame([{"prompt": "p", "completion": "p"}]).to_parquet(f"{TEST_DIR}/req2.parquet", index=False)
        
        manager = DataManager(task_type="sft")
        ds_json = manager.load_dataset(f"{TEST_DIR}/req2.json")
        ds_csv = manager.load_dataset(f"{TEST_DIR}/req2.csv")
        ds_par = manager.load_dataset(f"{TEST_DIR}/req2.parquet")
        
        passed = (len(ds_json["train"]) == 1 and len(ds_csv["train"]) == 1 and len(ds_par["train"]) == 1)
        print_result("Multi-Format Support (JSON, CSV, Parquet)", passed, "All loaded successfully")
    except Exception as e:
        print_result("Multi-Format Support", False, str(e))

    # -------------------------------------------------------------------------
    # REQUIREMENT 3: Local directory and HuggingFace dataset loading
    # -------------------------------------------------------------------------
    try:
        # Local Dir
        os.makedirs(f"{TEST_DIR}/local_data", exist_ok=True)
        with open(f"{TEST_DIR}/local_data/data.json", "w") as f: json.dump([{"prompt": "l", "completion": "l"}], f)
        
        manager = DataManager(task_type="sft")
        
        # Test 1: Local Dir
        ds_local = manager.load_dataset(f"{TEST_DIR}/local_data")
        
        # Test 2: HuggingFace (Alpaca) - Taking small slice
        try:
            ds_hf = manager.load_dataset("tatsu-lab/alpaca", split="train[:10]")
            hf_ok = True
        except:
            hf_ok = False # Network issues possible
            
        passed = len(ds_local["train"]) == 1 and hf_ok
        print_result("Local Directory & HF Loading", passed, f"Local: OK, HF: {'OK' if hf_ok else 'Skipped (Net)'}")
    except Exception as e:
        print_result("Local Directory & HF Loading", False, str(e))

    # -------------------------------------------------------------------------
    # REQUIREMENT 4: Custom column specification
    # -------------------------------------------------------------------------
    try:
        data = [{"my_weird_input": "Q", "my_weird_output": "A"}]
        with open(f"{TEST_DIR}/req4.json", "w") as f: json.dump(data, f)
        
        # Explicit mapping
        manager = DataManager(
            task_type="sft",
            column_mapping={"my_weird_input": "prompt", "my_weird_output": "completion"}
        )
        ds = manager.load_dataset(f"{TEST_DIR}/req4.json")
        
        passed = "prompt" in ds["train"].column_names
        print_result("Custom Column Specification", passed, f"Mapped: {ds['train'].column_names}")
    except Exception as e:
        print_result("Custom Column Specification", False, str(e))

    # -------------------------------------------------------------------------
    # REQUIREMENT 5: Dataset format transformation utilities (Task Schemas)
    # -------------------------------------------------------------------------
    try:
        # Testing DPO Transformation (user_input -> prompt, winner -> chosen)
        data = [{"user_input": "x", "winner": "y", "loser": "z"}]
        with open(f"{TEST_DIR}/req5.json", "w") as f: json.dump(data, f)
        
        manager = DataManager(task_type="dpo")
        ds = manager.load_dataset(f"{TEST_DIR}/req5.json")
        
        cols = ds["train"].column_names
        passed = "prompt" in cols and "chosen" in cols and "rejected" in cols
        print_result("Dataset Format Transformation (DPO)", passed, f"Transformed to: {cols}")
    except Exception as e:
        print_result("Dataset Format Transformation", False, str(e))

    # -------------------------------------------------------------------------
    # REQUIREMENT 6: System prompt injection capability
    # -------------------------------------------------------------------------
    try:
        data = [{"prompt": "Hello", "completion": "Hi"}]
        with open(f"{TEST_DIR}/req6.json", "w") as f: json.dump(data, f)
        
        manager = DataManager(task_type="sft", system_prompt="BE A PIRATE")
        ds = manager.load_dataset(f"{TEST_DIR}/req6.json")
        
        sample = ds["train"][0]["prompt"]
        passed = "BE A PIRATE" in sample
        print_result("System Prompt Injection", passed, f"Result: '{sample}'")
    except Exception as e:
        print_result("System Prompt Injection", False, str(e))

    # -------------------------------------------------------------------------
    # REQUIREMENT 7: Evaluation set generation
    # -------------------------------------------------------------------------
    try:
        # Create enough data to split (20 items)
        data = [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(20)]
        with open(f"{TEST_DIR}/req7.json", "w") as f: json.dump(data, f)
        
        manager = DataManager(task_type="sft", val_split_ratio=0.2)
        ds = manager.load_dataset(f"{TEST_DIR}/req7.json")
        
        has_train = "train" in ds
        has_val = "validation" in ds
        has_test = "test" in ds
        
        passed = has_train and has_val and has_test
        print_result("Evaluation Set Generation", passed, f"Splits: {list(ds.keys())}")
    except Exception as e:
        print_result("Evaluation Set Generation", False, str(e))

    # -------------------------------------------------------------------------
    # REQUIREMENT 8: Custom Processing Function (NEW!)
    # -------------------------------------------------------------------------
    try:
        data = [{"prompt": "abc", "completion": "123"}]
        with open(f"{TEST_DIR}/req8.json", "w") as f: json.dump(data, f)
        
        # Define custom function: Uppercase the prompt
        def uppercase_prompt(example):
            example["prompt"] = example["prompt"].upper()
            return example
            
        manager = DataManager(
            task_type="sft",
            processing_fn=uppercase_prompt
        )
        ds = manager.load_dataset(f"{TEST_DIR}/req8.json")
        
        result = ds["train"][0]["prompt"]
        passed = result == "ABC"
        print_result("Custom Processing Function", passed, f"Input: 'abc' -> Output: '{result}'")
    except Exception as e:
        print_result("Custom Processing Function", False, str(e))

    print("\n================================================================")
    teardown_env()

if __name__ == "__main__":
    main()