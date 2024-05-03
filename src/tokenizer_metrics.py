from abc import ABC, abstractmethod
from numpy import isnan, zeros
import pandas as pd
import regex as re
from statistics import mean, median
import pyrootutils
from scipy.stats import spearmanr
from transformers import AutoTokenizer


PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


def remove_tokenizer_formatting(s: str|list[str]) -> str|list[str]:
    if type(s) == list:
        formatted = [remove_tokenizer_formatting(x) for x in s]
        return [x for x in formatted if x is not None]
    if s[0] == "Ġ":
        return s[1:] if len(s) > 1 else None
    return s


WORDLIST_FILE = PROJECT_ROOT / "data/references/wordlist.txt"
MORPHEME_FILE = PROJECT_ROOT / "data/references/sigmorphon_morphemes.txt"
AOA_FIT_FILE = PROJECT_ROOT / "data/references/aoa_ws_fit.csv"
SIGMORPHON_DEV_FILE = PROJECT_ROOT / "data/references/sigmorphon_dev.tsv"


class SingleTokenizerMetric(ABC):
    """Metric for a single tokenizer."""
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
    

    @abstractmethod
    def calculate(self) -> float:
        ...


    def get_words_from_file(self, word_file):
        words = []
        with open(word_file, 'r') as f:
            words = [w.strip().lower() for w in f.readlines()]
        return set(words)


class MultiTokenizerMetric(ABC):
    """Metric that compares n tokenizers."""
    def __init__(self, tokenizers: list[AutoTokenizer]) -> None:
        self.tokenizers = tokenizers
        self.n = len(self.tokenizers)


    @abstractmethod
    def calculate(self):
        ...


class AverageTokenLength(SingleTokenizerMetric):
    """Mean/median token length of all tokens in this tokenizer."""
    def __init__(self, tokenizer: AutoTokenizer, metric: str|None="mean") -> None:
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
    how aligned whole-word token acquisition is to human CDI rates. """
    def __init__(self, tokenizers: list[AutoTokenizer], cdi_csv_file: str = AOA_FIT_FILE) -> None:
        super().__init__(tokenizers)
        self.cdi_aoa = self.format_cdi_file(cdi_csv_file)


    def format_cdi_file(self, cdi_file_name):
        aoa_dict = {}
        df = pd.read_csv(cdi_file_name, index_col=False)
        aoa_pattern = r'([\w/ ]+)(?:\*|!|$| \([\w ]+\))'
        item_names = [re.match(aoa_pattern, item).group(1) for item in list(df.item_definition)]
        for (item, aoa) in zip(item_names, df.aoa):
            # TODO: figure out how to deal with multi-word expressions, e.g. "belly button"
            if " " in item:
                continue
            if isnan(aoa): # remove items for which there is no predicted child AoA
                continue
            for expression in item.split("/"): # split on multiple expressions, e.g. "owie/boo boo"
                aoa_dict[expression] = aoa
        return aoa_dict

    
    def get_aoas(self) -> list[dict[str, (int, int)], list[str]]:
        tokenizer_aoa = {}
        remaining_cdi_words = set(self.cdi_aoa.keys())
        for i, tokenizer in enumerate(self.tokenizers):
            tokenized_words = tokenizer.encode_batch(list(remaining_cdi_words), add_special_tokens=False)
            successfully_tokenized = [(tokenized_word.tokens[0], tokenized_word.ids[0]) for tokenized_word in tokenized_words if len(tokenized_word.ids) == 1]
            tokenizer_aoa.update({word: (id, i) for (word, id) in successfully_tokenized})
            successfully_tokenized_words = {remove_tokenizer_formatting(word) for (word, _) in successfully_tokenized}
            remaining_cdi_words = remaining_cdi_words.difference(successfully_tokenized_words)
        return (tokenizer_aoa, remaining_cdi_words)

        
    def calculate(self):        
        # TODO: account for CDI words that have not been tokenized as one unit
        tokenizer_aoa, remaining_cdi_words = self.get_aoas()

        aoa_comparisons = [[tokenizer_aoa[word][1], self.cdi_aoa[remove_tokenizer_formatting(word)]] for word in tokenizer_aoa.keys()]
        # TODO: Use other rank metric?
        (coeff, pval) = spearmanr(aoa_comparisons)
        print (coeff, pval)
        self.pval = pval
        return coeff
        

class TokenizerOverlap(MultiTokenizerMetric):
    """Degree of overlap between n tokenizers."""
    def __init__(self, tokenizers: list[AutoTokenizer]) -> None:
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
    def __init__(self, tokenizer: AutoTokenizer, word_file: str = WORDLIST_FILE) -> None:
        super().__init__(tokenizer)
        self.word_list = self.get_words_from_file(word_file)

    
    def calculate(self) -> float:
        tokens = set(self.tokenizer.get_vocab().keys())
        overlap = self.word_list.intersection(tokens)
        return len(overlap) / len(tokens)
    

class CorrespondenceWithMorphemes(SingleTokenizerMetric):
    """How many tokens correspond with an English morpheme
        Using morphemes from the SIGMORPHON Shared Task 2022 + word list."""
    def __init__(self, tokenizer: AutoTokenizer, morpheme_file: str = MORPHEME_FILE, word_file: str = WORDLIST_FILE) -> None:
        super().__init__(tokenizer)
        self.word_list = self.get_words_from_file(morpheme_file)
        self.word_list.update(self.get_words_from_file(word_file))

    
    def calculate(self) -> float:
        tokens = set(self.tokenizer.get_vocab().keys())
        overlap = self.word_list.intersection(tokens)
        return len(overlap) / len(tokens)


class SplitsIntoMorphemes(SingleTokenizerMetric):
    """How different the tokenization of the dev split is from the gold split.
        count: how many words are split into the same number of morphemes as their gold split.
        distance: Levenshtein distance between the two tokenizations (e.g. token|iza|tion)"""
    def __init__(self, tokenizer: AutoTokenizer, 
                 sigmorphon_dev: str = SIGMORPHON_DEV_FILE, metric: str|None="count") -> None:
        super().__init__(tokenizer)
        self.words_and_morphs = self.get_morpheme_counts(sigmorphon_dev)
        self.metric = metric

    
    def calculate(self) -> float:
        words, gold_morphs = map(list, zip(*self.words_and_morphs))
        tokenized_words = self.tokenizer.encode_batch(list(words), add_special_tokens=False)
        if self.metric == "count":
            same_morphs = [len(x.ids) == len(y) for x,y in zip(tokenized_words, gold_morphs)]
            return sum(same_morphs) * 1.0 / len(words)
        if self.metric == "distance":
            gold_morph_strings = ["|".join(morphs) for morphs in gold_morphs]
            tokenized_strings = ["|".join(remove_tokenizer_formatting(x.tokens)) for x in tokenized_words]
            distances = [self.distance(tokenized, gold) for tokenized, gold in zip(tokenized_strings, gold_morph_strings)]
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
                if str1[y-1] == str2[x-1]:
                    dg = 0
                else:
                    dg = 1
                m[x, y] = min(m[x - 1, y] + 1, m[x, y - 1] + 1, m[x - 1, y - 1] + dg)
        return m[len(str2), len(str1)]


    def get_morpheme_counts(self, sigmorphon_dev_file) -> list[(str, list[str])]:
        counts = []
        with open(sigmorphon_dev_file, 'r') as f:
            for line in f:
                word, morphs, _ = line.split("\t")
                counts.append((word.strip(), morphs.replace("@@","").split(" ")))
        return counts