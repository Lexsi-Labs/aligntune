from setuptools import setup, find_packages

setup(
    name="tokenizer_extension",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.44.0",
        "tokenizers>=0.20.0",
        "datasets>=2.21.0",
        "numpy",
        "pandas",
        "tqdm",
        "requests",
        "sentencepiece",
        "fire",
        "protobuf>=3.20.0",
        "heapdict"
    ],
    extras_require={
        "full": ["PyICU", "tokenization-scorer", "conllu", "dask", "tiktoken>=0.7.0",],
    },
)
