import os
from itertools import islice

import requests
from urllib.parse import urlparse

import dask.dataframe as dd
from datasets import load_dataset


def download_file(url):
    parsed_url = urlparse(url)
    local_path = os.path.basename(parsed_url.path)
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the content to a local file
        with open(local_path, "wb") as file:
            file.write(response.content)
        return local_path

    raise ValueError(f"Failed to download the file. Status code: {response.status_code}")


def load_udtreebank_words(file_path: str):
    import conllu

    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    sentences = conllu.parse(data)

    return [token["form"] for sentence in sentences for token in sentence]


def load_et_edt_udtreebank_words(split="test"):
    file_url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_Estonian-EDT/master/et_edt-ud-{split}.conllu"
    path = download_file(file_url)
    return load_udtreebank_words(path)


def load_flores(lang, split="devtest"):
    dataset = load_dataset("facebook/flores", lang, trust_remote_code=True)[split]
    return [example["sentence"] for example in dataset]


def load_starcoder(n=1000, token=None):
    ds = load_dataset("bigcode/starcoderdata", data_dir="python", split="train", streaming=True)
    return [x["content"] for x in islice(ds, n)]


def load_mc4_valid(lang="et", n=None):
    df = dd.read_json(f"hf://datasets/allenai/c4/multilingual/c4-{lang}-validation.*.json.gz")
    return list(islice(df["text"], n))


def load_hf_dataset(name, split=None, subset=None, text_field="text", n=None):
    ds = load_dataset(name, split=split, subset=subset, streaming=True)
    return [x[text_field] for x in islice(ds, n)]
