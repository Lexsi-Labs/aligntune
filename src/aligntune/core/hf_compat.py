"""Compatibility shims for model architectures transformers doesn't fully cover."""

import importlib
import logging

logger = logging.getLogger(__name__)

# (model_type, model package, pretrained base class, new class name)
_SEQUENCE_CLASSIFICATION_GAPS = [
    ("cohere", "transformers.models.cohere", "CoherePreTrainedModel", "CohereForSequenceClassification"),
    ("cohere2", "transformers.models.cohere2", "Cohere2PreTrainedModel", "Cohere2ForSequenceClassification"),
]


def register_missing_sequence_classification() -> None:
    """Give Cohere-family text models an AutoModelForSequenceClassification class.

    transformers ships no CohereForSequenceClassification / Cohere2ForSequenceClassification,
    so reward and PPO value models built from Aya Expanse / Tiny Aya fail with
    "Unrecognized configuration class". Build them from GenericForSequenceClassification,
    the same base transformers uses for Llama/Qwen, and register them. Idempotent.

    AutoModelForSequenceClassification.register() is a silent no-op for config classes
    defined inside transformers (it guards against remote code hijacking native models),
    so this declares the class the way transformers declares its own: an entry in the
    model-type -> class-name table plus the class exposed on the model package.
    """
    try:
        from transformers.modeling_layers import GenericForSequenceClassification
        from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
    except ImportError:
        return

    for model_type, module_path, pretrained_name, cls_name in _SEQUENCE_CLASSIFICATION_GAPS:
        if model_type in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES:
            continue  # natively supported, or already registered
        try:
            module = importlib.import_module(module_path)
            pretrained_cls = getattr(module, pretrained_name)
        except (ImportError, AttributeError):
            continue
        if not hasattr(module, cls_name):
            model_cls = type(cls_name, (GenericForSequenceClassification, pretrained_cls), {"__module__": module_path})
            setattr(module, cls_name, model_cls)
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES[model_type] = cls_name
        logger.debug("Registered %s for AutoModelForSequenceClassification", cls_name)
