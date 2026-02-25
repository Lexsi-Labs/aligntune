"""
Authentication utilities for HuggingFace Hub integration.
"""

import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import login, whoami, HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Please install it with: pip install huggingface_hub")


def setup_hf_auth(token: Optional[str] = None, write_permission: bool = False) -> bool:
    """
    Setup HuggingFace authentication.
    
    Args:
        token: HuggingFace token. If None, will try to use existing token or prompt for login.
        write_permission: Whether to request write permission.
        
    Returns:
        bool: True if authentication successful, False otherwise.
    """
    if not HF_HUB_AVAILABLE:
        logger.error("huggingface_hub not available. Cannot setup authentication.")
        return False
    
    try:
        # Check if already authenticated
        try:
            user_info = whoami()
            if user_info:
                logger.info(f"Already authenticated as: {user_info['name']}")
                return True
        except Exception:
            pass  # Not authenticated yet
        
        # Determine token non-interactively
        effective_token = token or get_hf_token()
        if not effective_token:
            logger.info("No HuggingFace token found in arguments or environment; skipping login (non-interactive mode)")
            return False
        
        # Non-interactive login with token only
        login(token=effective_token, write_permission=write_permission)
        logger.info("Successfully authenticated with provided/environment token")
        
        # Verify authentication
        user_info = whoami()
        if user_info:
            logger.info(f"Authenticated as: {user_info['name']}")
            return True
        else:
            logger.error("Authentication verification failed")
            return False
            
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        return False


def get_hf_token() -> Optional[str]:
    """
    Get HuggingFace token from environment or file.
    
    Returns:
        str or None: The HuggingFace token if found.
    """
    # Try environment variable first
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    
    # Try to read from default HF cache directory
    try:
        hf_cache_dir = Path.home() / ".cache" / "huggingface"
        token_file = hf_cache_dir / "token"
        
        if token_file.exists():
            with open(token_file, 'r') as f:
                token = f.read().strip()
                if token:
                    return token
    except Exception as e:
        logger.debug(f"Could not read token from file: {e}")
    
    return None


def check_hf_auth() -> bool:
    """
    Check if HuggingFace authentication is setup.
    
    Returns:
        bool: True if authenticated, False otherwise.
    """
    if not HF_HUB_AVAILABLE:
        return False
    
    try:
        user_info = whoami()
        return user_info is not None
    except Exception:
        return False


def get_user_info() -> Optional[dict]:
    """
    Get current HuggingFace user information.
    
    Returns:
        dict or None: User information if authenticated.
    """
    if not HF_HUB_AVAILABLE:
        return None
    
    try:
        return whoami()
    except Exception:
        return None


def logout_hf():
    """Logout from HuggingFace Hub."""
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub not available")
        return
    
    try:
        # Remove token file
        hf_cache_dir = Path.home() / ".cache" / "huggingface"
        token_file = hf_cache_dir / "token"
        
        if token_file.exists():
            token_file.unlink()
            logger.info("Successfully logged out from HuggingFace Hub")
        else:
            logger.info("No active HuggingFace session found")
            
    except Exception as e:
        logger.error(f"Error during logout: {e}")


def test_hf_connection() -> bool:
    """
    Test HuggingFace Hub connection.
    
    Returns:
        bool: True if connection successful.
    """
    if not HF_HUB_AVAILABLE:
        logger.error("huggingface_hub not available")
        return False
    
    try:
        api = HfApi()
        # Try to get info about a public model
        model_info = api.model_info("gpt2")
        if model_info:
            logger.info("HuggingFace Hub connection successful")
            return True
        else:
            logger.error("Failed to connect to HuggingFace Hub")
            return False
    except Exception as e:
        logger.error(f"HuggingFace Hub connection failed: {e}")
        return False


def setup_git_config(name: Optional[str] = None, email: Optional[str] = None):
    """
    Setup git configuration for HuggingFace Hub operations.
    
    Args:
        name: Git user name
        email: Git user email
    """
    try:
        import subprocess
        
        if name:
            subprocess.run(["git", "config", "--global", "user.name", name], check=True)
            logger.info(f"Set git user.name to: {name}")
        
        if email:
            subprocess.run(["git", "config", "--global", "user.email", email], check=True)
            logger.info(f"Set git user.email to: {email}")
            
    except Exception as e:
        logger.warning(f"Could not setup git config: {e}")


# CLI helper function
def interactive_hf_setup():
    """Interactive setup for HuggingFace authentication."""
    print("=== AlignTune HuggingFace Setup ===")
    
    # Check current status
    if check_hf_auth():
        user_info = get_user_info()
        print(f"✓ Already authenticated as: {user_info['name']}")
        return True
    
    print("Setting up HuggingFace authentication...")
    
    # Test connection first
    if not test_hf_connection():
        print("❌ Cannot connect to HuggingFace Hub. Please check your internet connection.")
        return False
    
    # Try to get token from environment
    token = get_hf_token()
    if token:
        print("Found HuggingFace token in environment.")
        if setup_hf_auth(token=token):
            print("✓ Authentication successful!")
            return True
        else:
            print("❌ Failed to authenticate with found token.")
    
    # Interactive login
    print("Please login to HuggingFace Hub...")
    try:
        if setup_hf_auth():
            print("✓ Authentication successful!")
            return True
        else:
            print("❌ Authentication failed.")
            return False
    except KeyboardInterrupt:
        print("\n❌ Authentication cancelled by user.")
        return False