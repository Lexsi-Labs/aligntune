"""aligntune.data — data utilities (decontamination, etc.)."""

__all__ = ["decontaminate", "clean_dataset", "DeconReport"]


def __getattr__(name):
    if name in __all__:
        from aligntune.data import decontamination

        return getattr(decontamination, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
