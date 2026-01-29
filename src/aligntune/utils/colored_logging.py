"""
Colored logging and banner utilities for AlignTune.

Provides colored console output and ASCII art banners for better UX.
"""

import sys
import logging
from typing import Optional

# Try to import colorama for cross-platform colored output
try:
    from colorama import init, Fore, Back, Style, deinit
    COLORAMA_AVAILABLE = True
    # Initialize colorama (only needed on Windows, but safe to call on all platforms)
    init(autoreset=True)
except ImportError:
    COLORAMA_AVAILABLE = False
    # Create dummy color classes if colorama is not available
    class Fore:
        BLACK = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET = ""
    
    class Back:
        BLACK = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET = ""
    
    class Style:
        DIM = ""
        NORMAL = ""
        BRIGHT = ""
        RESET_ALL = ""


# =============================================================================
# ASCII ART BANNERS
# =============================================================================

ALIGNTUNE_BANNER = r"""
    _    _ _             _____                  
   / \  | (_) __ _ _ __ |_   _|   _ _ __   ___  
  / _ \ | | |/ _` | '_ \  | || | | | '_ \ / _ \ 
 / ___ \| | | (_| | | | | | || |_| | | | |  __/ 
/_/   \_\_|_|\__, |_| |_| |_| \__,_|_| |_|\___| 
             |___/                              
"""

ALIGNTUNE_EVALUATION_BANNER = r"""
    _    _ _             _____                       _____ __     __    _      _      
   / \  | (_) __ _ _ __ |_   _|   _ _ __   ___       | ____|\ \   / /   / \    | |    
  / _ \ | | |/ _` | '_ \  | || | | | '_ \ / _ \      |  _|   \ \ / /   / _ \   | |    
 / ___ \| | | (_| | | | | | || |_| | | | |  __/      | |___   \ V /   / ___ \  | |___ 
/_/   \_\_|_|\__, |_| |_| |_| \__,_|_| |_|\___|      |_____|   \_/   /_/   \_\ |_____|
             |___/                                                                     
"""


def print_aligntune_banner(subtitle: Optional[str] = None):
    """Print the AlignTune ASCII art banner with optional subtitle."""
    print("\n" + "=" * 80)
    print(Fore.CYAN + ALIGNTUNE_BANNER + Fore.RESET)
    if subtitle:
        print(Fore.YELLOW + subtitle + Fore.RESET)
    print("=" * 80 + "\n")


def print_section_banner(title: str, char: str = "=", width: int = 80, color: str = Fore.CYAN, use_ascii: bool = False):
    """Print a section banner with colored title."""
    if use_ascii and "EVALUATION" in title.upper():
        # Use ASCII art for ALIGNTUNE EVALUATION banner
        print("\n" + "=" * 80)
        print(Fore.CYAN + ALIGNTUNE_EVALUATION_BANNER + Fore.RESET)
        print("=" * 80 + "\n")
    else:
        print("\n" + color + char * width + Fore.RESET)
        print(color + f"  {title}".center(width) + Fore.RESET)
        print(color + char * width + Fore.RESET + "\n")


def print_subsection(title: str, char: str = "-", width: int = 80, color: str = Fore.BLUE):
    """Print a subsection header."""
    print(color + f"{title}".center(width, char) + Fore.RESET + "\n")


# =============================================================================
# COLORED LOGGING
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }
    
    RESET = Fore.RESET
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # Format the message
        return super().format(record)


def setup_colored_logging(
    logger_name: str = "aligntune",
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up colored logging for AlignTune.
    
    Args:
        logger_name: Name of the logger (default: "aligntune")
        level: Logging level (default: INFO)
        format_string: Custom format string (default: includes colors)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Set format
    if format_string is None:
        if COLORAMA_AVAILABLE:
            format_string = (
                f"{Fore.BLUE}[aligntune]{Fore.RESET} "
                f"{Style.DIM}%(asctime)s{Style.RESET_ALL} "
                f"%(levelname)s "
                f"{Style.DIM}%(name)s:%(lineno)d{Style.RESET_ALL} - "
                f"%(message)s"
            )
        else:
            format_string = "[aligntune] %(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s"
    
    formatter = ColoredFormatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# COLORED PRINT FUNCTIONS
# =============================================================================

def aligntune_info(message: str, prefix: str = "[aligntune]"):
    """Print an info message with AlignTune prefix."""
    print(f"{Fore.BLUE}{prefix}{Fore.RESET} {Fore.GREEN}INFO{Fore.RESET} - {message}")


def aligntune_warning(message: str, prefix: str = "[aligntune]"):
    """Print a warning message with AlignTune prefix."""
    print(f"{Fore.BLUE}{prefix}{Fore.RESET} {Fore.YELLOW}WARNING{Fore.RESET} - {message}")


def aligntune_error(message: str, prefix: str = "[aligntune]"):
    """Print an error message with AlignTune prefix."""
    print(f"{Fore.BLUE}{prefix}{Fore.RESET} {Fore.RED}ERROR{Fore.RESET} - {message}")


def aligntune_success(message: str, prefix: str = "[aligntune]"):
    """Print a success message with AlignTune prefix."""
    print(f"{Fore.BLUE}{prefix}{Fore.RESET} {Fore.GREEN}✓{Fore.RESET} {message}")


def aligntune_step(step_num: int, total_steps: int, message: str, prefix: str = "[aligntune]"):
    """Print a step message with progress indicator."""
    print(f"{Fore.BLUE}{prefix}{Fore.RESET} {Fore.CYAN}[{step_num}/{total_steps}]{Fore.RESET} {message}")


# =============================================================================
# PROGRESS INDICATORS
# =============================================================================

def print_progress_bar(current: int, total: int, width: int = 50, prefix: str = "[aligntune]"):
    """Print a progress bar."""
    if total == 0:
        return
    
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    
    print(
        f"\r{Fore.BLUE}{prefix}{Fore.RESET} "
        f"{Fore.CYAN}[{bar}]{Fore.RESET} "
        f"{Fore.YELLOW}{percent*100:.1f}%{Fore.RESET} "
        f"({current}/{total})",
        end="",
        flush=True
    )
    
    if current == total:
        print()  # New line when complete


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_aligntune_logging(show_banner: bool = True, subtitle: Optional[str] = None):
    """
    Initialize AlignTune colored logging and optionally show banner.
    
    Args:
        show_banner: Whether to show the AlignTune banner
        subtitle: Optional subtitle to display below banner
    """
    if show_banner:
        print_aligntune_banner(subtitle)
    
    # Set up colored logging
    logger = setup_colored_logging("aligntune", logging.INFO)
    
    if not COLORAMA_AVAILABLE:
        logger.warning("colorama not available. Install with: pip install colorama")
    
    return logger


# Cleanup function for colorama (if needed)
def cleanup_colored_logging():
    """Clean up colorama if it was initialized."""
    if COLORAMA_AVAILABLE:
        try:
            deinit()
        except:
            pass
