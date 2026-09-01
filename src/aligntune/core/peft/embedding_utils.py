"""Utilities for configuring trainable embedding modules."""

from typing import Any, List


def resolve_embedding_modules(model: Any) -> List[Any]:
    """Return the model's distinct input embedding and output-head modules."""
    modules = []
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None and all(module is not existing for existing in modules):
            modules.append(module)

    if not modules:
        raise ValueError(
            "train_embeddings=True, but the model does not expose input or "
            "output embeddings."
        )
    return modules


def resolve_embedding_module_names(model: Any) -> List[str]:
    """Resolve embedding module objects to their names in the model."""
    embedding_modules = resolve_embedding_modules(model)
    names = [
        name
        for name, module in model.named_modules()
        if any(module is embedding_module for embedding_module in embedding_modules)
    ]

    if len(names) != len(embedding_modules):
        raise ValueError(
            "train_embeddings=True, but not all input/output embedding modules "
            "could be resolved from model.named_modules()."
        )
    return list(dict.fromkeys(names))


def configure_embedding_only_training(model: Any) -> List[str]:
    """Freeze the model except for its input embeddings and output head."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    modules = resolve_embedding_modules(model)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    return resolve_embedding_module_names(model)
