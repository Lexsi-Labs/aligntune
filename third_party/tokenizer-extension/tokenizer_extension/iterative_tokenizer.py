import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterator


@dataclass
class Symbol:
    token_id: int
    byte_length: int
    prev_idx: int
    next_idx: int

    def merge_with(self, other, new_id):
        self.token_id = new_id
        self.byte_length += other.byte_length
        self.next_idx = other.next_idx


class Word:
    def __init__(self, symbols: Optional[List[Tuple[int, int]]]):
        self.symbols: List[Symbol] = []

        if symbols is not None:
            for symbol in symbols:
                self.add(*symbol)

    def add(self, token_id: int, byte_len: int):
        if len(self.symbols) == 0:
            prev_idx, next_idx = -1, -1
        else:
            self.symbols[-1].next_idx = len(self.symbols)
            prev_idx, next_idx = len(self.symbols) - 1, -1
        self.symbols.append(Symbol(token_id=token_id, byte_length=byte_len, prev_idx=prev_idx, next_idx=next_idx))

    def merge_iter(self, merges: Dict[Tuple[int, int], Tuple[int, int]]) -> Optional[Tuple[Tuple[int, int], int]]:
        merge_queue = []

        # Initialize priority queue with plausible merges
        for i in range(len(self.symbols) - 1):
            pair = (self.symbols[i].token_id, self.symbols[i + 1].token_id)
            if pair in merges:
                rank, new_id = merges[pair]
                heapq.heappush(merge_queue, (rank, i, pair, new_id))

        # Process the highest priority merge
        while merge_queue:
            _, index, (id1, id2), new_id = heapq.heappop(merge_queue)

            current_symbol = self.symbols[index]

            if current_symbol.byte_length == 0:
                continue

            if current_symbol.next_idx == -1:
                continue

            right_symbol = self.symbols[current_symbol.next_idx]

            candidate_merge = merges.get((current_symbol.token_id, right_symbol.token_id), None)
            if candidate_merge is None or candidate_merge[1] != new_id:
                continue

            if right_symbol.byte_length == 0:
                continue

            if current_symbol.token_id != id1 or right_symbol.token_id != id2:
                continue

            # perform merge
            current_symbol.merge_with(right_symbol, new_id)

            # delete previous token
            right_symbol.byte_length = 0

            # update next token
            if right_symbol.next_idx > -1:
                self.symbols[right_symbol.next_idx].prev_idx = index

            merge_result = (id1, id2), new_id

            # Update the queue with new plausible merges
            if current_symbol.prev_idx >= 0:
                prev_pair = (self.symbols[current_symbol.prev_idx].token_id, current_symbol.token_id)
                if prev_pair in merges:
                    prev_rank, prev_new_id = merges[prev_pair]
                    heapq.heappush(merge_queue, (prev_rank, current_symbol.prev_idx, prev_pair, prev_new_id))

            if current_symbol.next_idx < len(self.symbols) and current_symbol.next_idx != -1:
                next_pair = (current_symbol.token_id, self.symbols[current_symbol.next_idx].token_id)
                if next_pair in merges:
                    next_rank, next_new_id = merges[next_pair]
                    heapq.heappush(merge_queue, (next_rank, index, next_pair, next_new_id))

            yield merge_result

        self.symbols = [s for s in self.symbols if s.byte_length > 0]
        return None

    def get_token_ids(self) -> List[int]:
        return [s.token_id for s in self.symbols if s.byte_length > 0]


class IterativeTokenizer:
    def __init__(
            self,
            vocab: Dict[str, int],
            merges: List[Tuple[str, str]],
            byte_fallback: bool = False,
            unk_token: Optional[str] = None,
            ignore_merges: bool = False
    ):
        self.vocab = vocab
        self.merges = merges
        self.byte_fallback = byte_fallback
        self.unk_token = unk_token
        self.ignore_merges = ignore_merges

        # Create the merge_map with priority based on rank
        self.merge_map = self.create_merge_map(vocab, merges)
        self.vocab_r = {idx: tok for tok, idx in vocab.items()}

    def id_to_token(self, idx: int):
        return self.vocab_r[idx]

    def merge_id_to_token(self, merge: Optional[Tuple[Tuple[int, int], int]]):
        if merge is None:
            return None
        return ((self.vocab_r[merge[0][0]], self.vocab_r[merge[0][1]]), self.vocab_r[merge[1]])

    def word_to_tokens(self, word: Word):
        return [self.id_to_token(x) for x in word.get_token_ids()]

    def create_merge_map(self, vocab: Dict[str, int], merges: List[Tuple[str, str]]) -> Dict[
        Tuple[int, int], Tuple[int, int]]:
        merge_map = {}
        for i, (a, b) in enumerate(merges):
            a_id = vocab.get(a)
            if a_id is None:
                raise ValueError(f"Merge token '{a}' not found in vocabulary")

            b_id = vocab.get(b)
            if b_id is None:
                raise ValueError(f"Merge token '{b}' not found in vocabulary")

            new_token = f"{a}{b}"
            new_id = vocab.get(new_token)
            if new_id is None:
                raise ValueError(f"Merge token '{new_token}' not found in vocabulary")

            merge_map[(a_id, b_id)] = (i, new_id)

        return merge_map

    def tokenize_iteratively(self, word: str) -> Iterator[Tuple[Word, Optional[Tuple[Tuple[int, int], int]]]]:
        if self.ignore_merges and word in self.vocab:
            yield Word([(self.vocab.get(word), len(word))]), None
            return

        # Initialize tokens
        initial_symbols = []
        unk_id = self.vocab.get(self.unk_token, None)

        for char in word:
            char_id = self.vocab.get(char)
            if char_id is not None:
                initial_symbols.append((char_id, len(char)))
            elif self.byte_fallback:
                raise NotImplementedError("Byte fallback not implemented")
                # byte_symbols = [(self.vocab.get(f"<{b:02X}>", unk_id if unk_id is not None else b), 1) for b in char.encode('utf-8')]
                # initial_symbols.extend(byte_symbols)
            else:
                raise NotImplementedError("Unk token not implemented")
                # initial_symbols.append((unk_id if unk_id is not None else ord(char), len(char.encode('utf-8'))))

        word_obj = Word(initial_symbols)

        yield word_obj, None  # Yield the initial state

        for merge_result in word_obj.merge_iter(self.merge_map):
            if merge_result is None:
                continue
            yield word_obj, merge_result
