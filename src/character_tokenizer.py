"""CharacterTokenzier for Hugging Face Transformers.

This is heavily inspired from CanineTokenizer in transformers package.
"""

from enum import StrEnum
from typing import Sequence

from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

# from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class SpecialTokens(StrEnum):
    """Special tokens for tokenizer."""

    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    SEP = "[SEP]"
    CLS = "[CLS]"
    PAD = "[PAD]"
    MASK = "[MASK]"

    @classmethod
    def values(cls):
        """Return a list of the string values of each special token."""
        return list(map(lambda c: c.value, cls))

    @classmethod
    def as_dict(cls):
        """Return the special token as a dictionary."""
        return {v: i for i, v in enumerate(cls.values())}

    @property
    def index(self):
        """Return the index of the token in the vocabulary.

        Used to get the index of the PAD token when directly modifying tensors.
        """
        return SpecialTokens.values().index(self.value)


class CharacterTokenizer(PreTrainedTokenizerFast):
    def __init__(self, characters: Sequence[str], model_max_length: int, **kwargs):
        """Character tokenizer for Hugging Face transformers.

        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by [UNK].

            model_max_length (int): Model maximum sequence length.
        """

        vocab_dict = SpecialTokens.as_dict() | {
            ch: i + len(SpecialTokens.values()) for i, ch in enumerate(characters)
        }
        tokenizer_base = Tokenizer(
            WordLevel(vocab=vocab_dict, unk_token=SpecialTokens.UNK)
        )
        tokenizer_base.normalizer = normalizers.Sequence(
            [NFD(), StripAccents(), Lowercase()]
        )
        tokenizer_base.pre_tokenizer = Whitespace()
        tokenizer_base.post_processor = TemplateProcessing(
            single=f"{SpecialTokens.BOS} $A",
            special_tokens=[
                (SpecialTokens.BOS, tokenizer_base.token_to_id(SpecialTokens.BOS)),
            ],
        )

        super().__init__(
            tokenizer_object=tokenizer_base,
            bos_token=SpecialTokens.BOS,
            eos_token=SpecialTokens.EOS,
            unk_token=SpecialTokens.UNK,
            pad_token=SpecialTokens.EOS,  # use [EOS] as [PAD]
            mask_token=SpecialTokens.MASK,
            sep_token=SpecialTokens.SEP,
            cls_token=SpecialTokens.CLS,
            model_max_length=model_max_length,
            padding_side="left",
            **kwargs,
        )

        self.add_tokens(list(characters))
