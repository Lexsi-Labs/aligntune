import json
import warnings
from typing import Dict, List, Tuple, Optional

from tokenizers import Tokenizer, AddedToken
from tqdm import tqdm


def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        return json.load(file)


def write_json(data, file_path, indent=4):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=indent)


def get_vocab_and_merges(tokenizer) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    cfg = json.loads(tokenizer._tokenizer.to_str())
    merges = [tuple(x) if isinstance(x, list) else tuple(x.split(" ")) for x in cfg["model"]["merges"]]
    vocab = cfg["model"]["vocab"]
    return vocab, merges


def get_ordered_vocab(vocab) -> List[str]:
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    return [x[0] for x in sorted_vocab]


def replace_tokenizer_vocab_merges(tokenizer, vocab, merges):
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    tokenizer_json["model"]["merges"] = [" ".join(m) for m in merges]
    tokenizer_json["model"]["vocab"] = vocab
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
    return tokenizer


def budget_iterator(ds, limit=1_000_000_000):
    count = 0
    with tqdm(total=limit, miniters=limit // 100) as pbar:
        for x in ds:
            text = x["text"]
            count += len(text)
            if count >= limit:
                return
            pbar.update(len(text))
            yield text
    raise ValueError(f"The dataset too small to fill the budget ({count} / {limit}).")


def batch_iterator(dataset, batch_size=1000, verbose=False):
    if verbose:
        print(f"Dataset size: {len(dataset)}")

    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]


def update_postprocessor_special_tokens(hf_tokenizer, special_tokens_map=None):
    tokenizer_json = json.loads(hf_tokenizer._tokenizer.to_str())
    tokenizer = hf_tokenizer._tokenizer

    post_processor = tokenizer_json.pop("post_processor")
    if post_processor is not None:
        trained_tokenizer_json = json.loads(tokenizer.to_str())
        # Almost done, we just have to adjust the token IDs in the post processor
        _post_processors = [post_processor]
        # if the post-processor is of type Sequence, handle the individual processors
        if "processors" in post_processor:
            _post_processors.extend(post_processor["processors"])

        for _post_processor in _post_processors:
            if "special_tokens" in _post_processor:
                for key in _post_processor["special_tokens"]:
                    tokens = _post_processor["special_tokens"][key]["tokens"]
                    if special_tokens_map is not None:
                        tokens = [special_tokens_map.get(token, token) for token in tokens]
                    _post_processor["special_tokens"][key]["tokens"] = tokens
                    _post_processor["special_tokens"][key]["ids"] = [tokenizer.token_to_id(token) for token in tokens]

            for special_token in ["cls", "sep"]:
                if special_token in _post_processor:
                    token, _ = _post_processor[special_token]
                    if special_tokens_map is not None and token in special_tokens_map:
                        token = special_tokens_map[token]
                    token_id = tokenizer.token_to_id(token)
                    _post_processor[special_token] = [token, token_id]

        trained_tokenizer_json["post_processor"] = post_processor
        new_tokenizer = Tokenizer.from_str(json.dumps(trained_tokenizer_json))
    hf_tokenizer._tokenizer.post_processor = new_tokenizer.post_processor
    return hf_tokenizer


def get_added_tokens(tokenizer, special_only=False) -> List[Dict]:
    cfg = json.loads(tokenizer._tokenizer.to_str())
    return [x for x in cfg["added_tokens"] if not special_only or x["special"]]


def get_added_tokens_vocab(tokenizer, special_only=False) -> Dict[str, int]:
    return {x["content"]: x["id"] for x in get_added_tokens(tokenizer, special_only=special_only)}


def get_added_tokens_class(tokenizer, special_only=False) -> List[AddedToken]:
    return [AddedToken(**{k: v for k, v in x.items() if k != "id"}) for x in
            get_added_tokens(tokenizer, special_only=special_only)]


class override_ignore_merges:
    def __init__(self, tokenizer, new_value: Optional[bool]):
        self.tokenizer = tokenizer
        self.overide_value = new_value
        self.has_ignore_merges = hasattr(tokenizer._tokenizer.model, "ignore_merges")

        if self.has_ignore_merges:
            self.original_value = tokenizer._tokenizer.model.ignore_merges
        else:
            warnings.warn("The tokenizer does not support ignore_merges. The context manager will have no effect.")
            self.original_value = None

    def change_value(self, new_value: Optional[bool]):
        if self.has_ignore_merges and new_value is not None:
            self.tokenizer._tokenizer.model.ignore_merges = new_value

    def __enter__(self):
        self.change_value(self.overide_value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.change_value(self.original_value)


class disable_ignore_merges(override_ignore_merges):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, new_value=False)
