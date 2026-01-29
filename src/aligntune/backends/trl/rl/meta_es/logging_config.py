# logging_config.py
# Adaptive logging configuration for LLM reinforcement learning training


def get_adaptive_logging_config(max_steps):
    """
    Adaptive logging strategy based on training length.

    Provides reasonable defaults for logging_steps, save_steps, and save_total_limit
    that scale with the total training steps to balance monitoring, evaluation needs,
    and disk space usage.

    Args:
        max_steps (int): Total number of training steps

    Returns:
        tuple: (logging_steps, save_steps, save_total_limit)
            - logging_steps: How often to log metrics (steps)
            - save_steps: How often to save checkpoints (steps)
            - save_total_limit: Maximum number of checkpoints to keep on disk
    """
    # Logging frequency: Monitor training progress without spam
    # - Short runs (≤100 steps): Log every step
    # - Medium runs (1000 steps): Log every 10 steps
    # - Long runs (5000 steps): Log every 50 steps
    # - Very long runs (10000+ steps): Cap at 50 steps
    logging_steps = max(1, min(max_steps // 100, 50))

    # Save frequency: Aim for 15-20 checkpoints across training for evaluation
    # - 100 steps: Save every 5 steps (20 checkpoints)
    # - 1000 steps: Save every 50 steps (20 checkpoints)
    # - 5000 steps: Save every 250 steps (20 checkpoints)
    save_steps = max(1, max_steps // 20)

    # Checkpoint retention: Balance evaluation needs with disk space
    if max_steps <= 500:
        save_total_limit = 10  # Short runs: limit to 10 checkpoints
    elif max_steps <= 2000:
        save_total_limit = 15  # Medium runs: increase to 15 checkpoints
    else:
        save_total_limit = 20  # Long runs: keep 20 checkpoints for thorough evaluation

    return logging_steps, save_steps, save_total_limit


def print_logging_config(
        max_steps,
        logging_steps,
        save_steps,
        save_total_limit):
    """
    Print a summary of the adaptive logging configuration.

    Args:
        max_steps (int): Total training steps
        logging_steps (int): Logging frequency
        save_steps (int): Save frequency
        save_total_limit (int): Maximum checkpoints to keep
    """
    total_logs = max_steps // logging_steps
    total_saves = max_steps // save_steps

    print("=" * 60)
    print("📊 ADAPTIVE LOGGING CONFIGURATION")
    print("=" * 60)
    print(f"Training steps: {max_steps}")
    print(f"Logging every: {logging_steps} steps ({total_logs} total logs)")
    print(f"Saving every: {save_steps} steps ({total_saves} total saves)")
    print(f"Checkpoints kept: {save_total_limit} (disk space management)")
    print("=" * 60)
