"""
Base classes for the callback system.
"""

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


@dataclass
class TrainerControl:
    """Control flow of the training loop."""
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False


class TrainerState:
    """Base class for trainer state (compatibility wrapper)."""
    pass


class TrainerCallback(ABC):
    """
    Abstract base class for callbacks.
    
    All callbacks should inherit from this class and implement the methods
    they need. Methods not implemented will be ignored.
    """
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called at the end of the trainer initialization."""
        pass
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        pass
        
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        pass
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch."""
        pass
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of an epoch."""
        pass
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of a training step."""
        pass
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step."""
        pass
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation."""
        pass
        
    def on_save(self, args, state, control, **kwargs):
        """Called after saving a checkpoint."""
        pass
        
    def on_log(self, args, state, control, **kwargs):
        """Called after logging."""
        pass
        
    def on_prediction_step(self, args, state, control, **kwargs):
        """Called after a prediction step."""
        pass


class CallbackHandler:
    """
    Class to handle list of callbacks and manage control flow.
    """
    
    def __init__(self, callbacks: List[TrainerCallback], model, tokenizer, optimizer=None, scheduler=None):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(f"Callback {cb_class} is already present, skipping.")
        else:
            self.callbacks.append(cb)
            
    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                tokenizer=self.tokenizer,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                **kwargs,
            )
            # If callback returns a control object, update the current control object
            if result is not None:
                control = result
        return control

    def on_init_end(self, args, state, control):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args, state, control):
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args, state, control):
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(self, args, state, control):
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args, state, control):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args, state, control):
        return self.call_event("on_step_begin", args, state, control)

    def on_step_end(self, args, state, control):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args, state, control, metrics=None):
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_save(self, args, state, control):
        return self.call_event("on_save", args, state, control)

    def on_log(self, args, state, control, logs=None):
        return self.call_event("on_log", args, state, control, logs=logs)
        
    def on_prediction_step(self, args, state, control, **kwargs):
        """Called after a prediction step."""
        return self.call_event("on_prediction_step", args, state, control)
