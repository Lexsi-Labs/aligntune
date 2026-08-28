from setuptools import setup, find_packages
import sys

# mergekit is vendored (with two compatibility patches, see
# third_party/mergekit/PATCH_NOTES.md) rather than installed as a separate
# distribution, so it ships as part of the aligntune wheel/editable install.
# Its package list is fixed (matches third_party/mergekit/pyproject.toml)
# rather than auto-discovered, since it lives under a different root than
# aligntune's own src/ tree and declarative [tool.setuptools.packages.find]
# only supports one `where` root.
_MERGEKIT_PACKAGES = [
    "mergekit",
    "mergekit.io",
    "mergekit.merge_methods",
    "mergekit.moe",
    "mergekit.scripts",
    "mergekit.evo",
    "mergekit.tokenizer",
    "mergekit.tokensurgeon",
    "mergekit.architecture",
    "mergekit._data",
    "mergekit._data.architectures",
    "mergekit._data.chat_templates",
]

# unsloth and unsloth_zoo are vendored the same way (see
# third_party/unsloth/PATCH_NOTES.md and third_party/unsloth_zoo/PATCH_NOTES.md
# for the AGPL-3.0-licensed files included and why) rather than installed as
# separate distributions, so `pip install aligntune` alone is enough - no separate
# `--no-deps unsloth==...` step. Package lists are auto-discovered (unlike
# mergekit's fixed list above) since each lives under its own third_party
# root and has no name collisions with the aligntune/mergekit trees.
_UNSLOTH_PACKAGES = find_packages(where="third_party/unsloth")
_UNSLOTH_ZOO_PACKAGES = find_packages(where="third_party/unsloth_zoo")

if __name__ == "__main__":
    if sys.version_info < (3, 10):
        raise RuntimeError("AlignTune requires Python 3.10 or higher")
    # Most metadata is deferred to pyproject.toml (PEP 621); this stub adds
    # only the package discovery that can't be expressed declaratively there.
    setup(
        packages=(
            find_packages(where="src")
            + _MERGEKIT_PACKAGES
            + _UNSLOTH_PACKAGES
            + _UNSLOTH_ZOO_PACKAGES
        ),
        package_dir={
            "": "src",
            "mergekit": "third_party/mergekit/mergekit",
            "unsloth": "third_party/unsloth/unsloth",
            "unsloth_zoo": "third_party/unsloth_zoo/unsloth_zoo",
        },
    )