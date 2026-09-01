import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional, List, Dict

from tqdm import tqdm

from tokenizer_extension.utils import get_ordered_vocab

try:
    import icu
except ImportError:
    icu = None

START_SYMBOL = '▁'
SPECIAL = {'<unk>', '<s>', '</s>', '<pad>', '<mask>'}
BYTE_VOCAB = {'<0x00>', '<0x01>', '<0x02>', '<0x03>', '<0x04>', '<0x05>', '<0x06>', '<0x07>', '<0x08>', '<0x09>',
              '<0x0A>', '<0x0B>', '<0x0C>', '<0x0D>', '<0x0E>', '<0x0F>', '<0x10>', '<0x11>', '<0x12>', '<0x13>',
              '<0x14>', '<0x15>', '<0x16>', '<0x17>', '<0x18>', '<0x19>', '<0x1A>', '<0x1B>', '<0x1C>', '<0x1D>',
              '<0x1E>', '<0x1F>', '<0x20>', '<0x21>', '<0x22>', '<0x23>', '<0x24>', '<0x25>', '<0x26>', '<0x27>',
              '<0x28>', '<0x29>', '<0x2A>', '<0x2B>', '<0x2C>', '<0x2D>', '<0x2E>', '<0x2F>', '<0x30>', '<0x31>',
              '<0x32>', '<0x33>', '<0x34>', '<0x35>', '<0x36>', '<0x37>', '<0x38>', '<0x39>', '<0x3A>', '<0x3B>',
              '<0x3C>', '<0x3D>', '<0x3E>', '<0x3F>', '<0x40>', '<0x41>', '<0x42>', '<0x43>', '<0x44>', '<0x45>',
              '<0x46>', '<0x47>', '<0x48>', '<0x49>', '<0x4A>', '<0x4B>', '<0x4C>', '<0x4D>', '<0x4E>', '<0x4F>',
              '<0x50>', '<0x51>', '<0x52>', '<0x53>', '<0x54>', '<0x55>', '<0x56>', '<0x57>', '<0x58>', '<0x59>',
              '<0x5A>', '<0x5B>', '<0x5C>', '<0x5D>', '<0x5E>', '<0x5F>', '<0x60>', '<0x61>', '<0x62>', '<0x63>',
              '<0x64>', '<0x65>', '<0x66>', '<0x67>', '<0x68>', '<0x69>', '<0x6A>', '<0x6B>', '<0x6C>', '<0x6D>',
              '<0x6E>', '<0x6F>', '<0x70>', '<0x71>', '<0x72>', '<0x73>', '<0x74>', '<0x75>', '<0x76>', '<0x77>',
              '<0x78>', '<0x79>', '<0x7A>', '<0x7B>', '<0x7C>', '<0x7D>', '<0x7E>', '<0x7F>', '<0x80>', '<0x81>',
              '<0x82>', '<0x83>', '<0x84>', '<0x85>', '<0x86>', '<0x87>', '<0x88>', '<0x89>', '<0x8A>', '<0x8B>',
              '<0x8C>', '<0x8D>', '<0x8E>', '<0x8F>', '<0x90>', '<0x91>', '<0x92>', '<0x93>', '<0x94>', '<0x95>',
              '<0x96>', '<0x97>', '<0x98>', '<0x99>', '<0x9A>', '<0x9B>', '<0x9C>', '<0x9D>', '<0x9E>', '<0x9F>',
              '<0xA0>', '<0xA1>', '<0xA2>', '<0xA3>', '<0xA4>', '<0xA5>', '<0xA6>', '<0xA7>', '<0xA8>', '<0xA9>',
              '<0xAA>', '<0xAB>', '<0xAC>', '<0xAD>', '<0xAE>', '<0xAF>', '<0xB0>', '<0xB1>', '<0xB2>', '<0xB3>',
              '<0xB4>', '<0xB5>', '<0xB6>', '<0xB7>', '<0xB8>', '<0xB9>', '<0xBA>', '<0xBB>', '<0xBC>', '<0xBD>',
              '<0xBE>', '<0xBF>', '<0xC0>', '<0xC1>', '<0xC2>', '<0xC3>', '<0xC4>', '<0xC5>', '<0xC6>', '<0xC7>',
              '<0xC8>', '<0xC9>', '<0xCA>', '<0xCB>', '<0xCC>', '<0xCD>', '<0xCE>', '<0xCF>', '<0xD0>', '<0xD1>',
              '<0xD2>', '<0xD3>', '<0xD4>', '<0xD5>', '<0xD6>', '<0xD7>', '<0xD8>', '<0xD9>', '<0xDA>', '<0xDB>',
              '<0xDC>', '<0xDD>', '<0xDE>', '<0xDF>', '<0xE0>', '<0xE1>', '<0xE2>', '<0xE3>', '<0xE4>', '<0xE5>',
              '<0xE6>', '<0xE7>', '<0xE8>', '<0xE9>', '<0xEA>', '<0xEB>', '<0xEC>', '<0xED>', '<0xEE>', '<0xEF>',
              '<0xF0>', '<0xF1>', '<0xF2>', '<0xF3>', '<0xF4>', '<0xF5>', '<0xF6>', '<0xF7>', '<0xF8>', '<0xF9>',
              '<0xFA>', '<0xFB>', '<0xFC>', '<0xFD>', '<0xFE>', '<0xFF>'}


@dataclass
class TrainerSpec:
    allow_whitespace_only_pieces: bool = True
    treat_whitespace_as_suffix: bool = False
    split_by_whitespace: bool = True
    split_by_number: bool = True
    split_digits: bool = True
    split_by_unicode_script: bool = True
    max_sentencepiece_length: int = 16


SEPARATE_TOKENS = SPECIAL.union(BYTE_VOCAB)


def is_valid_merge(t1, t2, cfg: TrainerSpec):
    if t1 in SEPARATE_TOKENS or t2 in SEPARATE_TOKENS:
        return False
    if len(t1) == 0 or len(t2) == 0:
        raise ValueError("Received a zero-length piece for merge")

    token = t1 + t2
    if len(token) > cfg.max_sentencepiece_length:
        return False

    all_whitespace_piece = all(t == START_SYMBOL for t in token)
    prev_script = None
    for pos, c in enumerate(token):
        if c == START_SYMBOL:
            if not cfg.allow_whitespace_only_pieces or not all_whitespace_piece:
                if cfg.treat_whitespace_as_suffix:
                    if (cfg.split_by_whitespace and pos < len(token) - 1) or (
                            not cfg.split_by_whitespace and pos < len(token) - 1 and pos == 0):
                        return False
                elif (cfg.split_by_whitespace and pos > 0) or (
                        not cfg.split_by_whitespace and pos > 0 and pos == len(token) - 1):
                    return False
        else:
            if cfg.split_digits and c.isdigit():
                return False

            script = icu.Script.getScript(c).getName()
            if script == 'Hiragana' or script == 'Katakana':
                script = 'Han'
            if script == 'Inherited':
                script = prev_script

            if not cfg.split_by_number and c.isdigit():
                script = None

            if cfg.split_by_unicode_script and script is not None and prev_script is not None and script != prev_script:
                return False
            prev_script = script
    return True


def group_tokens(text, tokenizer, **sp_kwargs):
    tokens = tokenizer.tokenize(text)
    cfg = TrainerSpec(**sp_kwargs)
    if len(tokens) == 0:
        return []
    grouped_tokens = []
    group = [tokens[0]]
    for token in tokens[1:]:
        if is_valid_merge(group[-1], token, cfg):
            group.append(token)
        else:
            if len(group) > 0:
                grouped_tokens.append(group)
            group = [token]

    if len(group) > 0:
        grouped_tokens.append(group)

    return list(map(tuple, grouped_tokens))


def get_script(char: str, prev: Optional[str] = None) -> str:
    if icu is None:
        raise ImportError('This function requires PyICU.')
    script = icu.Script.getScript(char).getName()
    if script == 'Hiragana' or script == 'Katakana':
        return 'Han'
    if script == 'Inherited' and prev is not None:
        return prev
    return script


def get_token_script(token: str, prev: Optional[str] = None):
    if len(token) == 0:
        raise ValueError("Empty token.")

    if token == START_SYMBOL:
        return None

    if len(token) == 1:
        return get_script(token, prev)

    if token[0] == START_SYMBOL:
        start = 1
    else:
        start = 0

    script = get_script(token[start], prev)
    for x in token[start + 1:]:
        current = get_script(x, prev=script)
        if script != current:
            print("Different scripts in one token", token)
        script = current

    return script


def read_model_proto(path: str):
    from sentencepiece.sentencepiece_model_pb2 import ModelProto
    model = ModelProto()
    with open(path, "rb") as f:
        model.ParseFromString(f.read())
    return model


def save_model_proto(model, model_path: str):
    with open(model_path, "wb") as f:
        f.write(model.SerializeToString())


def save_vocab_proto(model, vocab_path: str):
    with open(vocab_path, "w", encoding="utf-8") as f:
        for p in model.pieces:
            f.write(f"{p.piece}\t{int(p.score)}\n")


def train_sentencepiece_from_model(
        tokenizer_path: str,
        input_path: str,
        vocab_size: int = 64000,
        model_prefix: str = "sp_model",
        num_threads: int = 16,
):
    import sentencepiece as spm

    model = read_model_proto(tokenizer_path)
    config = {x[0].name: getattr(model.trainer_spec, x[0].name) for x in model.trainer_spec.ListFields()}
    assert config["model_type"] == 2
    kwargs = {
        k: config[k]
        for k in [
            'character_coverage', 'input_sentence_size', 'seed_sentencepiece_size', 'shrinking_factor',
            'num_sub_iterations', 'max_sentence_length', 'shuffle_input_sentence', 'max_sentencepiece_length',
            'split_by_unicode_script', 'split_by_whitespace', 'split_by_number', 'treat_whitespace_as_suffix',
            'split_digits', 'allow_whitespace_only_pieces', 'vocabulary_output_piece_score', 'hard_vocab_limit',
            'use_all_vocab', 'byte_fallback'
        ]
    }

    print(f"Training SentencePiece model with the following parameters: {kwargs}", flush=True)

    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        num_threads=num_threads,
        model_type="bpe",
        **kwargs
    )


ILLEGAL_CHARS = {" ", "\n", "\r", "", "\t"}


def read_sentencepiece_from_vocab(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.split()[0] for line in f]


def read_sentencepiece_vocab_from_model(path: str) -> list:
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor(model_file=path)
    return [tokenizer.id_to_piece(idx) for idx in range(tokenizer.vocab_size())]


def read_sentencepiece_vocab(path: str) -> list:
    if path.endswith(".model"):
        return read_sentencepiece_vocab_from_model(path)
    return read_sentencepiece_from_vocab(path)


def train_coverage_extension(
        data: List[str],
        vocab: Dict[str, int],
        coverage: float = 0.9995,
        max_tokens: Optional[int] = None,
):
    counter = Counter()
    for doc in tqdm(data):
        counter.update(doc.replace(" ", "▁"))

    for c in ILLEGAL_CHARS:
        del counter[c]

    total_characters = sum(counter.values())
    logging.info(f"Total characters: {total_characters}")
    in_vocab_characters = sum(c for k, c in counter.items() if k in vocab)
    initial_coverage = in_vocab_characters / total_characters
    logging.info(f"The vocabulary coverage is {initial_coverage}")

    if initial_coverage >= coverage:
        logging.info(f"Coverage already achieved, no need to extend")
        return {}

    sorted_characters = sorted(
        [(k, c) for k, c in counter.items() if k not in vocab],
        key=lambda x: x[1],
        reverse=True
    )
    n_tokens = 0
    new_characters = in_vocab_characters
    for _, c in sorted_characters:
        new_characters += c
        n_tokens += 1
        if new_characters / total_characters >= coverage:
            break

    logging.info(f"Number of tokens to add: {n_tokens} (coverage={new_characters / total_characters})")

    if max_tokens is not None and n_tokens > max_tokens:
        logging.info(f"Number of tokens to add exceeds max_tokens, capping at {max_tokens}")
        n_tokens = max_tokens

    sorted_characters = sorted_characters[:n_tokens]
    new_coverage = (sum(x for _, x in sorted_characters) + in_vocab_characters) / total_characters
    logging.info(f"New coverage: {new_coverage}")

    return {x[0]: idx for idx, x in enumerate(sorted_characters)}


# Reorder vocab so that character vocabulary gets added first
def reorder_sp_vocab(vocab: dict) -> dict:
    token_list = get_ordered_vocab(vocab)
    base_character_idxs = [i for i, x in enumerate(token_list) if len(x) == 1]
    min_idx = min(base_character_idxs)
    max_idx = max(base_character_idxs)

    if not all(len(x) == 1 for x in token_list[min_idx:max_idx + 1]):
        print("Detected range of characters is not continuous")

    base_vocab = [x for x in token_list if len(x) == 1]
    base_tokens = set(base_vocab)
    new_vocab = base_vocab + [x for x in token_list if x not in base_tokens]
    return {piece: idx for idx, piece in enumerate(new_vocab)}
