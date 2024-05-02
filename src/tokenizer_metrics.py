from abc import ABC, abstractmethod
from numpy import isnan
import pandas as pd
import regex as re
from statistics import median
from scipy.stats import spearmanr
from transformers import AutoTokenizer


class SingleTokenizerMetric(ABC):
    """Metric for a single tokenizer."""
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
    

    @abstractmethod
    def calculate(self) -> float:
        ...


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
            return sum(item_lengths) / len(item_lengths)
        else:
            pass



class AlignmentWithCDI(MultiTokenizerMetric):
    """Given n tokenizers representing increasing subsets, calculate
    how aligned whole-word token acquisition is to human CDI rates. """
    def __init__(self, tokenizers: list[AutoTokenizer], cdi_csv_file: str) -> None:
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
            successfully_tokenized_words = {self.make_compatible_with_cdi_tokens(word) for (word, _) in successfully_tokenized}
            remaining_cdi_words = remaining_cdi_words.difference(successfully_tokenized_words)
        return (tokenizer_aoa, remaining_cdi_words)


    def make_compatible_with_cdi_tokens(self, s: str) -> str:
        if s[0] == "Ġ":
            return s[1:]
        return s

        
    def calculate(self):        
        # TODO: account for CDI words that have not been tokenized as one unit
        tokenizer_aoa, remaining_cdi_words = self.get_aoas()

        aoa_comparisons = [[tokenizer_aoa[word][1], self.cdi_aoa[self.make_compatible_with_cdi_tokens(word)]] for word in tokenizer_aoa.keys()]
        print(f"AOAs: {aoa_comparisons}")
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
        return len(overlap) / len(total)


class CorrespondenceWithWords(SingleTokenizerMetric):
    """How many tokens in the tokenizer correspond to an English word."""
    def __init__(self, tokenizer: AutoTokenizer, word_file: str) -> None:
        super().__init__(tokenizer)
        self.word_list = self.get_words_from_file(word_file)


    def get_words_from_file(self, word_file):
        words = []
        with open(word_file, 'r') as f:
            words = [w.strip().lower() for w in f.readlines()]
        return set(words)

    
    def calculate(self) -> float:
        tokens = set(self.tokenizer.get_vocab().keys())
        overlap = self.word_list.intersection(tokens)
        return len(overlap) / len(tokens)
    

class CorrespondenceWithMorphemes(SingleTokenizerMetric):
    """How many tokens correspond with an English morpheme
    Using morphemes from this list?: https://education.ufl.edu/patterson/files/2020/05/Morphemes-and-Their-Meanings.pdf
    """
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        pass

    
    def calculate(self) -> float:
        pass