from typing import Optional, Iterable, List

from .base import StaticPrunerBase, register_pruner
from tokenizer_extension.utils import get_ordered_vocab

try:
    import icu
except ImportError:
    icu = None


@register_pruner("script")
class ScriptPruner(StaticPrunerBase):
    def __init__(
            self,
            allowed_scripts: Optional[Iterable[str]] = None,
            forbidden_scripts: Optional[Iterable[str]] = None
    ):
        super().__init__()
        if icu is None:
            raise ImportError("icu module is required for script pruning")
        self.allowed_scripts = set(allowed_scripts) if allowed_scripts is not None else None
        self.forbidden_scripts = set(forbidden_scripts) if forbidden_scripts is not None else None

    @staticmethod
    def icu_script_filter(text: str, allowed_scripts=None, forbidden_scripts=None) -> bool:
        scripts = [icu.Script.getScript(x).getName() for x in text]

        if allowed_scripts is not None and not all([script in allowed_scripts for script in scripts]):
            return False

        if forbidden_scripts is not None and any([script in forbidden_scripts for script in scripts]):
            return False

        return True

    def calculate_pruning_order(self, tokenizer, training_data=None) -> List[str]:
        vocab = tokenizer.get_vocab()
        tokens_to_remove = [
            x for x in reversed(get_ordered_vocab(vocab))
            if not self.icu_script_filter(
                tokenizer.decode(vocab[x]),
                allowed_scripts=self.allowed_scripts,
                forbidden_scripts=self.forbidden_scripts
            )
        ]
        return tokens_to_remove


@register_pruner("latin_script")
class LatinScriptPruner(ScriptPruner):
    def __init__(self):
        super().__init__(allowed_scripts={"Latin", "Common", "Inherited"})


@register_pruner("latin_cyrillic_script")
class LatinCyrillicScriptPruner(ScriptPruner):
    def __init__(self):
        super().__init__(allowed_scripts={"Cyrillic", "Latin", "Common", "Inherited"})
