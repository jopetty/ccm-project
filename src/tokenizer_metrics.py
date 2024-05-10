"""Metrics for tokenizers."""

from abc import ABC, abstractmethod
from statistics import mean, median

import pandas as pd
import pyrootutils
import regex as re
from numpy import isnan, zeros
from scipy.stats import spearmanr
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents
from tqdm import tqdm

from data import normalize_string

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


WORDLIST_FILE = PROJECT_ROOT / "data/references/wordlist.txt"
MORPHEME_FILE = PROJECT_ROOT / "data/references/sigmorphon_morphemes.txt"
AOA_FIT_FILE = PROJECT_ROOT / "data/references/aoa_ws_fit.csv"
SIGMORPHON_DEV_FILE = PROJECT_ROOT / "data/references/sigmorphon_dev.tsv"
TEST_FILE = PROJECT_ROOT / "data/test/simple_wiki.test"


def remove_tokenizer_formatting(s: str | list[str]) -> str | list[str]:
    # check if s is a list
    # old_s = s
    if isinstance(s, list):
        formatted = [remove_tokenizer_formatting(x) for x in s]
        return [x for x in formatted if x is not None]
    if s[0] == "Ġ":
        return s[1:] if len(s) > 1 else None
    # print(f"{old_s} -> {s}")
    # raise SystemError
    return s


class SingleTokenizerMetric(ABC):
    """Metric for a single tokenizer."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def calculate(self) -> float: ...

    def get_words_from_file(self, word_file):
        words = []
        with open(word_file, "r") as f:
            words = [w.strip().lower() for w in f.readlines()]
        return set(words)


class MultiTokenizerMetric(ABC):
    """Metric that compares n tokenizers."""

    def __init__(self, tokenizers: list[Tokenizer]) -> None:
        self.tokenizers = tokenizers
        self.n = len(self.tokenizers)

    @abstractmethod
    def calculate(self): ...


class AverageTokenLength(SingleTokenizerMetric):
    """Mean/median token length of all tokens in this tokenizer."""

    def __init__(self, tokenizer: Tokenizer, metric: str | None = "mean") -> None:
        super().__init__(tokenizer)
        self.metric = metric if metric else "mean"

    def calculate(self) -> float:
        item_lengths = [len(k) for k in self.tokenizer.get_vocab()]
        # TODO: maybe add normalization for word-initial and word-medial tokens?
        if self.metric == "median":
            return float(median(item_lengths))
        elif self.metric == "mean":
            return mean(item_lengths)
        else:
            pass


class AlignmentWithCDI(MultiTokenizerMetric):
    """Given n tokenizers representing increasing subsets, calculate
    how aligned whole-word token acquisition is to human CDI rates."""

    def __init__(
        self, tokenizers: list[Tokenizer], cdi_csv_file: str = AOA_FIT_FILE
    ) -> None:
        super().__init__(tokenizers)
        self.cdi_aoa = self.format_cdi_file(cdi_csv_file)

    def format_cdi_file(self, cdi_file_name):
        aoa_dict = {}
        df = pd.read_csv(cdi_file_name, index_col=False)
        aoa_pattern = r"([\w/ ]+)(?:\*|!|$| \([\w ]+\))"
        item_names = [
            re.match(aoa_pattern, item).group(1) for item in list(df.item_definition)
        ]
        for item, aoa in zip(item_names, df.aoa):
            # TODO: figure out how to deal with multi-word expressions, e.g.
            # "belly button"
            if " " in item:
                continue
            if isnan(aoa):  # remove items for which there is no predicted child AoA
                continue
            for expression in item.split(
                "/"
            ):  # split on multiple expressions, e.g. "owie/boo boo"
                aoa_dict[expression] = aoa

        # lowercase all keys in aoa_dict
        aoa_dict = {k.lower(): v for k, v in aoa_dict.items()}
        return aoa_dict

    def get_aoas(self) -> list[dict[str, (int, int)], list[str]]:
        tokenizer_aoa = {}
        remaining_cdi_words = set(self.cdi_aoa.keys())
        for i, tokenizer in enumerate(self.tokenizers):
            # <<<<<<< fixes
            #             tokenized_words = tokenizer.encode_batch(
            #                 list(remaining_cdi_words), add_special_tokens=False
            #             )
            #             successfully_tokenized = [
            #                 (tokenized_word.tokens[0], tokenized_word.ids[0])
            #                 for tokenized_word in tokenized_words
            #                 if len(tokenized_word.ids) == 1
            #             ]
            #             tokenizer_aoa.update(
            #                 {word: (id, i) for (word, id) in successfully_tokenized}
            #             )
            #             successfully_tokenized_words = {
            #                 self.make_compatible_with_cdi_tokens(word)
            #                 for (word, _) in successfully_tokenized
            #             }
            #             remaining_cdi_words = remaining_cdi_words.difference(
            #                 successfully_tokenized_words
            #             )
            #         return (tokenizer_aoa, remaining_cdi_words)

            #     def make_compatible_with_cdi_tokens(self, s: str) -> str:
            #         if s[0] == "Ġ":
            #             return s[1:]
            #         return s

            #     def calculate(self):
            #  # TODO: account for CDI words that have not been tokenized as one unit
            #         tokenizer_aoa, remaining_cdi_words = self.get_aoas()

            #         aoa_comparisons = [
            #             [
            #                 tokenizer_aoa[word][1],
            #                 self.cdi_aoa[self.make_compatible_with_cdi_tokens(word)],
            #             ]
            #             for word in tokenizer_aoa.keys()
            #         ]
            #         print(f"AOAs: {aoa_comparisons}")
            # =======
            tokenized_words = tokenizer.encode_batch(
                list(remaining_cdi_words), add_special_tokens=False
            )
            successfully_tokenized = [
                (tokenized_word.tokens[0], tokenized_word.ids[0])
                for tokenized_word in tokenized_words
                if len(tokenized_word.ids) == 1
            ]
            tokenizer_aoa.update(
                {word: (id, i) for (word, id) in successfully_tokenized}
            )
            successfully_tokenized_words = {
                remove_tokenizer_formatting(word)
                for (word, _) in successfully_tokenized
            }
            remaining_cdi_words = remaining_cdi_words.difference(
                successfully_tokenized_words
            )
        return (tokenizer_aoa, remaining_cdi_words)

    def calculate(self):
        # TODO: account for CDI words that have not been tokenized as one unit
        tokenizer_aoa, remaining_cdi_words = self.get_aoas()

        aoa_comparisons = [
            [tokenizer_aoa[word][1], self.cdi_aoa[remove_tokenizer_formatting(word)]]
            for word in tokenizer_aoa.keys()
        ]
        # >>>>>>> main
        # TODO: Use other rank metric?
        (coeff, pval) = spearmanr(aoa_comparisons)
        print(coeff, pval)
        self.pval = pval
        return coeff


class TokenizerOverlap(MultiTokenizerMetric):
    """Degree of overlap between n tokenizers."""

    def __init__(self, tokenizers: list[Tokenizer]) -> None:
        super().__init__(tokenizers)

    def calculate(self):
        tokens = []
        for t in self.tokenizers:
            tokens.append(set(t.get_vocab().keys()))
        overlap = set.intersection(*tokens)
        total = set.union(*tokens)
        return len(overlap) * 1.0 / len(total)


class CorrespondenceWithWords(SingleTokenizerMetric):
    """How many tokens in the tokenizer correspond to an English word.
    Using words from https://github.com/dwyl/english-words/blob/master/words_alpha.txt"""

    def __init__(self, tokenizer: Tokenizer, word_file: str = WORDLIST_FILE) -> None:
        super().__init__(tokenizer)
        self.word_list = self.get_words_from_file(word_file)

    def get_words_from_file(self, word_file):
        words = []
        with open(word_file, "r") as f:
            words = [w.strip().lower() for w in f.readlines()]
        return set(words)

    def calculate(self) -> float:
        tokens = set(self.tokenizer.get_vocab().keys())
        overlap = self.word_list.intersection(tokens)
        return len(overlap) / len(tokens)


class CorrespondenceWithMorphemes(SingleTokenizerMetric):
    """How many tokens correspond with an English morpheme
    Using morphemes from the SIGMORPHON Shared Task 2022 + word list."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        morpheme_file: str = MORPHEME_FILE,
        word_file: str = WORDLIST_FILE,
    ) -> None:
        super().__init__(tokenizer)
        self.word_list = self.get_words_from_file(morpheme_file)
        self.word_list.update(self.get_words_from_file(word_file))

    def calculate(self) -> float:
        tokens = set(self.tokenizer.get_vocab().keys())
        overlap = self.word_list.intersection(tokens)
        return len(overlap) / len(tokens)


class SplitsIntoMorphemes(SingleTokenizerMetric):
    """How different the tokenization of the dev split is from the gold split.
    count: how many words are split into the same number of morphemes as their
    gold split.
    distance: Levenshtein distance between the two tokenizations
    (e.g. token|iza|tion)"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        sigmorphon_dev: str = SIGMORPHON_DEV_FILE,
        metric: str | None = "count",
    ) -> None:
        super().__init__(tokenizer)
        self.words_and_morphs = self.get_morpheme_counts(sigmorphon_dev)
        self.metric = metric

    def calculate(self) -> float:
        words, gold_morphs = map(list, zip(*self.words_and_morphs))
        tokenized_words = self.tokenizer.encode_batch(
            list(words), add_special_tokens=False
        )
        if self.metric == "count":
            same_morphs = [
                len(x.ids) == len(y) for x, y in zip(tokenized_words, gold_morphs)
            ]
            return sum(same_morphs) * 1.0 / len(words)
        if self.metric == "distance":
            gold_morph_strings = ["|".join(morphs) for morphs in gold_morphs]
            tokenized_strings = [
                "|".join(remove_tokenizer_formatting(x.tokens)) for x in tokenized_words
            ]
            distances = [
                self.distance(tokenized, gold)
                for tokenized, gold in zip(tokenized_strings, gold_morph_strings)
            ]
            return mean(distances)
        else:
            pass

    def distance(self, str1, str2) -> float:
        """Simple Levenshtein implementation.
        Taken from https://github.com/sigmorphon/2022SegmentationST/blob/main/evaluation/evaluate.py"""
        m = zeros([len(str2) + 1, len(str1) + 1], dtype=float)
        for x in range(1, len(str2) + 1):
            m[x, 0] = m[x - 1, 0] + 1
        for y in range(1, len(str1) + 1):
            m[0, y] = m[0, y - 1] + 1
        for x in range(1, len(str2) + 1):
            for y in range(1, len(str1) + 1):
                if str1[y - 1] == str2[x - 1]:
                    dg = 0
                else:
                    dg = 1
                m[x, y] = min(m[x - 1, y] + 1, m[x, y - 1] + 1, m[x - 1, y - 1] + dg)
        return m[len(str2), len(str1)]

    def get_morpheme_counts(self, sigmorphon_dev_file) -> list[(str, list[str])]:
        counts = []
        with open(sigmorphon_dev_file, "r") as f:
            for line in f:
                word, morphs, _ = line.split("\t")
                counts.append(
                    (word.strip().lower(), morphs.replace("@@", "").lower().split(" "))
                )
        return counts


class SplitsOnSpace(SingleTokenizerMetric):
    """Whether a tokenizer trained without spaces create tokenizations that
    coincide with word boundarieson a test set."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        baseline: str = "tokenized",
        test_file: str = TEST_FILE,
    ) -> None:
        super().__init__(tokenizer)
        self.test_sentences = self.read_text_file(test_file)
        self.normalizer = Sequence([NFD(), StripAccents(), Lowercase()])
        self.baseline = baseline
        self.pretokenizer = ByteLevel(add_prefix_space=False, use_regex=False)

    def calculate(self) -> float:
        spaces_kept = 0
        total_spaces = 0
        for line in tqdm(self.test_sentences):
            tokenized_line = remove_tokenizer_formatting(
                self.tokenizer.encode(line, add_special_tokens=False).tokens
            )
            tokenized = "|".join(tokenized_line).strip()
            space_split = re.sub(
                r"\s+", "|", self.normalizer.normalize_str(line)
            ).strip()
            if self.baseline == "gold":
                kept, total = self.check_spaces(tokenized, space_split)
            elif self.baseline == "tokenized":
                kept, total = self.check_spaces(space_split, tokenized)
            spaces_kept += kept
            total_spaces += total
        return spaces_kept * 1.0 / total_spaces

    def read_text_file(self, text_file):
        with open(text_file, "r") as f:
            lines = f.readlines()
            return [
                normalize_string(line.strip(), remove_all_spaces=False)
                for line in lines
            ]

    def check_spaces(self, test: str, baseline: str) -> list[int, int]:
        i = 0
        j = 0
        kept = 0
        total = 0
        while i < len(baseline) and j < len(test):
            if baseline[i] == "|":
                total += 1
                i += 1
                if test[j] == "|":
                    kept += 1
                    j += 1
            elif baseline[i] == test[j]:
                i += 1
                j += 1
            elif test[j] == "|":
                j += 1
            elif baseline[i] == "":
                i += 1
            else:
                # TODO: Currently we just break if the BPE decoding is too weird.
                # Fix if possible.
                print(baseline[:i] + "**" + baseline[i + 1 :])
                print(test[:j] + "**" + test[j + 1 :])
                break
        return [kept, total]
