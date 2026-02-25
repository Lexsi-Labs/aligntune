"""
Enhanced model loading utilities with support for local weights and various formats.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

# NOTE: Transformers and Unsloth imports are lazy-loaded inside methods
# to prevent "poisoning" the environment or consuming VRAM unnecessarily.

logger = logging.getLogger(__name__)


class ModelLoader:
    """Enhanced model loader with support for local weights and various configurations."""
    
    def __init__(self):
        self.device = self._detect_device()
        
    def _detect_device(self) -> str:
        """Detect the best available device."""
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_local_weights(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        device_map: Union[str, Dict] = "auto",
        torch_dtype: Optional[Any] = None, # Type Any to avoid early torch import
        trust_remote_code: bool = False,
        use_unsloth: bool = False,
        max_seq_length: int = 2048,
        load_in_4bit: bool = False
    ) -> Tuple[Any, Any]:
        """
        Load model and tokenizer from local weights.
        """
        model_path = Path(model_path)
        tokenizer_path = Path(tokenizer_path) if tokenizer_path else model_path
        config_path = Path(config_path) if config_path else model_path
        
        # Validate paths
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        logger.info(f"Loading model from local path: {model_path}")
        
        # Auto-detect model type and structure
        model_info = self._analyze_local_model(model_path)
        logger.info(f"Detected model info: {model_info}")
        
        try:
            # FIX: Robust Unsloth check. 
            can_use_unsloth = False
            if use_unsloth:
                try:
                    import unsloth
                    can_use_unsloth = True
                except ImportError:
                    logger.warning("Unsloth requested but not installed. Falling back to Transformers.")
            
            if can_use_unsloth:
                return self._load_with_unsloth(
                    str(model_path), max_seq_length, load_in_4bit, trust_remote_code
                )
            else:
                return self._load_with_transformers(
                    str(model_path), str(tokenizer_path), device_map, 
                    torch_dtype, trust_remote_code, load_in_4bit
                )
                
        except Exception as e:
            logger.error(f"Error loading local weights: {e}")
            raise
    
    def _analyze_local_model(self, model_path: Path) -> Dict[str, Any]:
        """Analyze local model structure and detect format."""
        model_info = {
            "path": str(model_path),
            "format": "unknown",
            "has_config": False,
            "has_tokenizer": False,
            "has_safetensors": False,
            "has_pytorch_bin": False,
            "files": []
        }
        
        if model_path.is_file():
            # Single file model
            model_info["format"] = "single_file"
            model_info["files"] = [model_path.name]
            
            if model_path.suffix == ".safetensors":
                model_info["has_safetensors"] = True
            elif model_path.suffix in [".bin", ".pt", ".pth"]:
                model_info["has_pytorch_bin"] = True
                
        elif model_path.is_dir():
            # Directory with model files
            model_info["format"] = "directory"
            files = list(model_path.iterdir())
            model_info["files"] = [f.name for f in files]
            
            # Check for specific files
            for file in files:
                if file.name == "config.json":
                    model_info["has_config"] = True
                elif file.name in ["tokenizer.json", "tokenizer_config.json"]:
                    model_info["has_tokenizer"] = True
                elif file.suffix == ".safetensors":
                    model_info["has_safetensors"] = True
                elif file.name.startswith("pytorch_model") and file.suffix == ".bin":
                    model_info["has_pytorch_bin"] = True
        
        return model_info
    
    def _load_with_unsloth(
        self, 
        model_path: str, 
        max_seq_length: int, 
        load_in_4bit: bool, 
        trust_remote_code: bool
    ) -> Tuple[Any, Any]:
        """Load model using Unsloth with robust fallbacks."""
        logger.info("Loading model with Unsloth")
        
        # Lazy imports
        from unsloth import FastLanguageModel
        from transformers import AutoTokenizer, AutoConfig
        
        # 1. Load Tokenizer Manually (Robustly)
        tokenizer = None
        config_obj = None
        try:
            # Pre-load config to ensure it's an object, not a dict
            config_obj = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            
            # --- AGGRESSIVE CONFIG SANITIZATION ---
            # Remove keys that cause Qwen2 init crashes when combined with Unsloth injection
            keys_to_remove = ["max_position_embeddings", "rope_scaling"]
            for key in keys_to_remove:
                if hasattr(config_obj, key):
                    delattr(config_obj, key)
                    logger.debug(f"Sanitized {key} from config to prevent Unsloth conflict.")

            # Load Tokenizer using the sanitized config
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                config=config_obj,
                trust_remote_code=trust_remote_code
            )
        except Exception as e:
            logger.warning(f"Manual tokenizer loading failed: {e}. Will let Unsloth try.")
            tokenizer = None
            # If manual config load failed, we can't pass a config object.
            config_obj = None

        # 2. Load Model via Unsloth with Global Tokenizer Patch
        try:
            # INTERCEPTION STRATEGY (GLOBAL PATCH):
            # Unsloth calls 'AutoTokenizer.from_pretrained' internally. In patched environments,
            # this crashes with 'dict object has no attribute model_type' if we pass a config object.
            # BUT we MUST pass the config object to prevent the Model crash (Qwen2 parameter conflict).
            #
            # Solution: We temporarily overwrite AutoTokenizer.from_pretrained to bypass Unsloth's
            # internal loading logic entirely and return our manually loaded tokenizer.
            
            # Save original method
            original_from_pretrained = AutoTokenizer.from_pretrained
            
            # Define the mock
            def mock_from_pretrained(*args, **kwargs):
                if tokenizer is not None:
                    return tokenizer
                # Fallback to original if we don't have a manual tokenizer
                return original_from_pretrained(*args, **kwargs)
            
            # Apply Patch
            AutoTokenizer.from_pretrained = mock_from_pretrained
            
            try:
                # Load Model
                # - fix_tokenizer=False: We handled it manually.
                # - config=config_obj: WE PASS THE SANITIZED CONFIG. This fixes the Model Crash.
                # - load_tokenizer=True (Default): Let Unsloth call our Mock. This fixes the Tokenizer Crash.
                model, unsloth_tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=max_seq_length,
                    dtype=None,
                    load_in_4bit=load_in_4bit,
                    trust_remote_code=trust_remote_code,
                    fix_tokenizer=False,
                    config=config_obj, 
                )
            finally:
                # Restore original method immediately
                AutoTokenizer.from_pretrained = original_from_pretrained

            # 3. Resolve Tokenizer
            if tokenizer is None:
                tokenizer = unsloth_tokenizer

        except Exception as e:
            logger.error(f"Unsloth loading failed: {e}")
            raise e
        
        # Ensure pad token is set
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def _load_with_transformers(
        self,
        model_path: str,
        tokenizer_path: str,
        device_map: Union[str, Dict],
        torch_dtype: Optional[Any],
        trust_remote_code: bool,
        load_in_4bit: bool,
        config: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """Load model using standard transformers."""
        logger.info("Loading model with transformers")
        
        # Lazy imports
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
        import torch
        
        # Set torch dtype default if None
        if torch_dtype is None:
            torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32

        # Explicitly load config if not provided
        if config is None:
            try:
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            except Exception:
                config = None

        # Load tokenizer with retries for different modes
        tokenizer = None
        tokenizer_error = None
        
        # Attempt 1: Standard load
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                config=config,
                trust_remote_code=trust_remote_code
            )
        except Exception as e:
            tokenizer_error = e
            # Attempt 2: Use fast=False (often fixes 'dict' attribute errors in patched envs)
            try:
                logger.info("Retrying tokenizer load with use_fast=False...")
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    config=config,
                    trust_remote_code=trust_remote_code,
                    use_fast=False
                )
            except Exception as e2:
                logger.warning(f"Could not load tokenizer: {e2}")
                # Attempt 3: Try loading from model path instead of tokenizer path
                if tokenizer_path != model_path:
                    logger.info("Attempting to load tokenizer from model path")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_path,
                            config=config,
                            trust_remote_code=trust_remote_code
                        )
                    except Exception:
                        pass
        
        if tokenizer is None:
            raise ValueError(f"Failed to load tokenizer. Last error: {tokenizer_error}")
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": device_map if self.device.startswith("cuda") else None,
        }
        
        model_kwargs["torch_dtype"] = torch_dtype
        
        # Add quantization config if requested
        # Check if BitsAndBytes is available (try import)
        try:
            import bitsandbytes
            bnb_available = True
        except ImportError:
            bnb_available = False

        if load_in_4bit and bnb_available:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs['quantization_config'] = quantization_config
        elif load_in_4bit:
            logger.warning("4-bit quantization requested but BitsAndBytesConfig not available")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # Resize token embeddings if needed
        if tokenizer.pad_token_id >= model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    
    def load_from_hub_or_local(
        self,
        model_name_or_path: str,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load model from HuggingFace Hub or local path automatically.
        """
        # Check if it's a local path
        if os.path.exists(model_name_or_path):
            logger.info(f"Detected local path: {model_name_or_path}")
            return self.load_local_weights(model_name_or_path, **kwargs)
        else:
            logger.info(f"Loading from HuggingFace Hub: {model_name_or_path}")
            return self._load_from_hub(model_name_or_path, **kwargs)
    
    def _load_from_hub(
        self,
        model_name: str,
        use_unsloth: bool = False,
        max_seq_length: int = 2048,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        device_map: Union[str, Dict] = "auto",
        torch_dtype: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """Load model from HuggingFace Hub."""
        # FIX: Robust Unsloth check for Hub loading too
        can_use_unsloth = False
        if use_unsloth:
            try:
                import unsloth
                can_use_unsloth = True
            except ImportError:
                pass

        if can_use_unsloth:
            return self._load_with_unsloth(
                model_name, max_seq_length, load_in_4bit, trust_remote_code
            )
        else:
            return self._load_with_transformers(
                model_name, model_name, device_map, torch_dtype, trust_remote_code, load_in_4bit
            )
    
    def convert_checkpoint_format(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        output_format: str = "safetensors"
    ):
        """Convert checkpoint between different formats."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        logger.info(f"Converting checkpoint from {input_path} to {output_path}")
        logger.info(f"Target format: {output_format}")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input checkpoint does not exist: {input_path}")
        
        try:
            # Load model
            model, tokenizer = self.load_local_weights(input_path)
            
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save in target format
            if output_format == "safetensors":
                model.save_pretrained(
                    output_path,
                    safe_serialization=True
                )
            elif output_format == "pytorch":
                model.save_pretrained(
                    output_path,
                    safe_serialization=False
                )
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            # Save tokenizer
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"Successfully converted checkpoint to {output_format} format")
            
        except Exception as e:
            logger.error(f"Error converting checkpoint: {e}")
            raise
    
    def get_model_info(self, model_path_or_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        # Lazy import
        from transformers import AutoTokenizer, AutoConfig
        
        info = {
            "path_or_name": model_path_or_name,
            "is_local": os.path.exists(model_path_or_name),
            "config": None,
            "tokenizer_info": None,
            "model_size": None,
            "architecture": None,
            "vocab_size": None
        }
        
        try:
            # Load config
            if info["is_local"]:
                config_path = Path(model_path_or_name) / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        info["config"] = json.load(f)
                else:
                    config = AutoConfig.from_pretrained(model_path_or_name)
                    info["config"] = config.to_dict()
            else:
                config = AutoConfig.from_pretrained(model_path_or_name)
                info["config"] = config.to_dict()
            
            # Extract key information from config
            if info["config"]:
                info["architecture"] = info["config"].get("architectures", ["Unknown"])[0]
                info["vocab_size"] = info["config"].get("vocab_size")
                info["model_size"] = self._estimate_model_size(info["config"])
            
            # Get tokenizer info
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
                info["tokenizer_info"] = {
                    "vocab_size": len(tokenizer),
                    "model_max_length": tokenizer.model_max_length,
                    "has_pad_token": tokenizer.pad_token is not None,
                    "special_tokens": {
                        "bos_token": tokenizer.bos_token,
                        "eos_token": tokenizer.eos_token,
                        "pad_token": tokenizer.pad_token,
                        "unk_token": tokenizer.unk_token
                    }
                }
            except Exception as e:
                logger.warning(f"Could not load tokenizer info: {e}")
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
        
        return info
    
    def _estimate_model_size(self, config: Dict) -> Optional[str]:
        """Estimate model size from config."""
        try:
            if "num_parameters" in config:
                return self._format_parameter_count(config["num_parameters"])
            
            # Estimate from architecture
            hidden_size = config.get("hidden_size", config.get("d_model", 0))
            num_layers = config.get("num_hidden_layers", config.get("num_layers", 0))
            vocab_size = config.get("vocab_size", 0)
            
            if hidden_size and num_layers and vocab_size:
                # Rough estimation
                embedding_params = vocab_size * hidden_size
                layer_params = num_layers * (12 * hidden_size * hidden_size)  # Approximate
                total_params = embedding_params + layer_params
                
                return self._format_parameter_count(total_params)
                
        except Exception:
            pass
        
        return None
    
    def _format_parameter_count(self, count: int) -> str:
        """Format parameter count in human readable form."""
        if count >= 1e9:
            return f"{count / 1e9:.1f}B"
        elif count >= 1e6:
            return f"{count / 1e6:.1f}M"
        elif count >= 1e3:
            return f"{count / 1e3:.1f}K"
        else:
            return str(count)
    
    def list_local_models(self, base_path: Union[str, Path] = "./models") -> List[Dict[str, Any]]:
        """List all local models in a directory."""
        base_path = Path(base_path)
        models = []
        
        if not base_path.exists():
            logger.warning(f"Base path does not exist: {base_path}")
            return models
        
        for item in base_path.iterdir():
            if item.is_dir():
                try:
                    model_info = self._analyze_local_model(item)
                    if model_info["has_config"] or model_info["has_safetensors"] or model_info["has_pytorch_bin"]:
                        models.append({
                            "name": item.name,
                            "path": str(item),
                            "info": model_info
                        })
                except Exception as e:
                    logger.debug(f"Error analyzing {item}: {e}")
        
        return models
    
    def cleanup_cache(self, cache_dir: Optional[Union[str, Path]] = None):
        """Clean up model cache directory."""
        import shutil
        
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "huggingface"
        else:
            cache_dir = Path(cache_dir)
        
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"Cleaned cache directory: {cache_dir}")
            except Exception as e:
                logger.error(f"Error cleaning cache: {e}")
        else:
            logger.info(f"Cache directory does not exist: {cache_dir}")


# Utility functions for backwards compatibility
def load_local_model(model_path: str, **kwargs) -> Tuple[Any, Any]:
    """Convenience function to load a local model."""
    loader = ModelLoader()
    return loader.load_local_weights(model_path, **kwargs)


def load_model_auto(model_name_or_path: str, **kwargs) -> Tuple[Any, Any]:
    """Convenience function to load model automatically from Hub or local path."""
    loader = ModelLoader()
    return loader.load_from_hub_or_local(model_name_or_path, **kwargs)


def get_model_info(model_path_or_name: str) -> Dict[str, Any]:
    """Convenience function to get model information."""
    loader = ModelLoader()
    return loader.get_model_info(model_path_or_name)