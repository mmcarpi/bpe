"""
    A faster Python implementation of Byte Pair Enconding algorithm.
    This implementation uses a linked list (llst.py) to represent the text for fast merges 
    and a max heap (heap.py) to track the most common pair.

    This code is based on Karpathy implementation minBPE, which is available at https://github.com/karpathy/minbpe/tree/master
"""

# Notice that regex is not the default regular expression engine
import regex as re

from pathlib import Path
from unicodedata import category as cat
from typing import Dict, List, Optional, Tuple
from collections import Counter

from llst import LinkedList
from heap import Heap

CURRENT_VERSION = "TOKENIZER V1"

# Utility functions for printing tokens based on minBPE


def replace_control_chars(s: str) -> str:
    s = "".join((f"\\u{ord(c):04x}" if cat(c)[0] == "C" else c) for c in s)
    return s


def render_token(t: bytes) -> str:
    s = t.decode("utf-8", errors="replace")
    s = replace_control_chars(s)
    return s


# Split patterns for tokenizer
SPLIT_PATTERNS = dict(
    default=r"""\p{L}+|\p{P}+|\p{N}|\p{S}|\p{Z}{1, 2, 4}""",
    gpt4=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
)


# Core BPE functions and variables


def insertpair(
    pairs: Dict[Tuple[int, int], Tuple[int, int]],
    pair: Tuple[int, int],
    idx: Tuple[int, int],
):
    if pair not in pairs:
        pairs[pair] = set([idx])
    else:
        pairs[pair].add(idx)


def get_stats(
    ids: List[List[int]], merges: Optional[Dict[Tuple[int, int], int]] = None
):
    pairs = {}
    freq = {}
    dlls = []
    for i, chunk_id in enumerate(ids):
        dlls.append(LinkedList.from_list(chunk_id))
        for idx, pair in enumerate(zip(chunk_id, chunk_id[1:])):
            insertpair(pairs, pair, (i, idx))
            freq[pair] = freq.get(pair, 0) + 1

    heap = Heap()
    for pair, freq in freq.items():
        if merges is None:
            heap.push(freq, pair)
        else:
            heap.push(-merges.get(pair, float("inf")), pair)

    return dlls, pairs, heap


def merge(
    pair: Tuple[int, int],
    nitem: int,
    dlls: List[LinkedList],
    pairs: Dict[Tuple[int, int], Tuple[int, int]],
):
    if pair not in pairs:
        return
    indices = pairs[pair]
    oldpairs = []
    newpairs = []
    while indices:
        idll, idx = indices.pop()
        dll = dlls[idll]
        node = dll.node[idx]

        if mpair := dll.pair_ending_at(idx):
            pairs[mpair[0]].remove((idll, mpair[1]))
            oldpairs.append(mpair[0])

        if mpair := dll.pair_starting_at(node.next):
            pairs[mpair[0]].remove((idll, mpair[1]))
            oldpairs.append(mpair[0])

        dll.pop(node.next)
        node.item = nitem

        if mpair := dll.pair_ending_at(idx):
            insertpair(pairs, mpair[0], (idll, mpair[1]))
            newpairs.append(mpair[0])

        if mpair := dll.pair_starting_at(idx):
            insertpair(pairs, mpair[0], (idll, mpair[1]))
            newpairs.append(mpair[0])

    return (Counter(newpairs), Counter(oldpairs))


def update_stats(heap: Heap, newpairs: Counter, oldpairs: Counter):
    for pair, freq in newpairs.items():
        heap.push(freq, pair)

    for pair, freq in oldpairs.items():
        heap.decrease(freq, pair)


class Tokenizer:

    def __init__(self, pattern=None):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.pattern: str = pattern or SPLIT_PATTERNS["default"]
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens = {}
        self.vocab: Dict[int, bytes] = self._build_vocab()

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def train(self, text: str, vocabsize: int, threshold: int = 1, verbose=False):
        assert vocabsize >= 256
        num_merges = vocabsize - 256

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(c.encode("utf-8")) for c in text_chunks]

        merges = dict()
        dlls, pairs, heap = get_stats(ids)
        for i in range(num_merges):

            if len(heap) == 0:
                if verbose:
                    print("train: no more pairs")
                break

            (key, top_pair) = heap.pop()

            if key <= threshold:
                if verbose:
                    print(f"train: top_pair occurs less than {threshold}")
                break

            idx = 256 + i
            if verbose:
                print("merge(ids, %s, %i)" % (top_pair, idx))
            merges[top_pair] = idx
            newpairs, oldpairs = merge(top_pair, idx, dlls, pairs)
            update_stats(heap, newpairs, oldpairs)

        if verbose:
            idslen = sum(map(len, ids))
            dllslen = sum(map(len, dlls))
            print(f"train: input length was {len(idslen)}. New length is {dllslen}")
            print(f"train: compression rate of {len(idslen)/dllslen:.2f}")

        self.merges = merges
        self.vocab = self._build_vocab()

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        dll, pairs, heap = get_stats(ids, self.merges)

        while len(ids) >= 2:
            key, pair = heap.pop()
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            merge(pair, idx, dll, pairs)
        return dll.to_list()

    def encode_ordinary(self, text):
        tokens = text.encode("utf-8")
        tokens = [int(tok) for tok in tokens]

        dlls, pairs, heap = get_stats([tokens])
        while True:
            (key, pair) = heap.pop()
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            merge(pair, idx, dlls, pairs)
        return dlls[0].to_list()

    def encode(self, text, allowed_special="none_raise"):
        def assert_none_raise(text):
            assert all(token not in text for token in self.special_tokens)

        params = {
            "all": lambda: self.special_tokens,
            "none": lambda: None,
            "none_raise": lambda: assert_none_raise(text),
        }

        if isinstance(allowed_special, set):
            special = {
                k: v for k, v in self.special_tokens.items() if k in allowed_special
            }
        else:
            special = params[allowed_special]()

        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _build_vocab(self) -> Dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, path: Path, name: str, save_vocab=True):
        model_file = path / (name + ".model")
        with open(model_file, "w") as f:
            f.write(f"{CURRENT_VERSION}\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        if save_vocab:
            vocab_file = path / (name + ".vocab")
            segrem = {idx: pair for pair, idx in self.merges.items()}
            with open(vocab_file, "w", encoding="utf-8") as f:
                for idx, token in self.vocab.items():
                    s = render_token(token)
                    if idx in segrem:
                        idx0, idx1 = segrem[idx]
                        s0 = render_token(self.vocab[idx0])
                        s1 = render_token(self.vocab[idx1])
                        f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                    else:
                        f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith(".model"), "Expected .model file!"

        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline.strip()
            assert (
                version == CURRENT_VERSION
            ), f"expected {CURRENT_VERSION} got {version}"
            self.pattern = f.readline().strip()
            num_special_tokens = int(f.readline().strip())
            for _ in range(num_special_tokens):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)

            for line in f:
                pair = map(int, line.split())
                merges[pair] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
