from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
from pathlib import Path
from pprint import pformat

from data import load_data, SpecialTokens
from datasets import Dataset
import fire
from itertools import pairwise
from math import exp
import pyrootutils
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFD, Lowercase, Replace, StripAccents
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence as PTSequence
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

class TokenizerTrainer(ABC):
    @property
    @abstractmethod
    def tokenizer_base(self) -> Tokenizer:
        return self._tokenizer_base

    def get_tokenizer(self) -> Tokenizer:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer_base(),
            bos_token=SpecialTokens.BOS,
            eos_token=SpecialTokens.EOS,
            unk_token=SpecialTokens.UNK,
            pad_token=SpecialTokens.EOS,  # use [EOS] as [PAD]
            mask_token=SpecialTokens.MASK,
            sep_token=SpecialTokens.SEP,
            cls_token=SpecialTokens.CLS,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer


    @abstractmethod
    def train(self, dataset: Dataset, initial_alphabet: list[str]):
        ...
    

class BPETokenizerTrainer(TokenizerTrainer):
    def __init__(self, vocab_size: int, min_frequency: int, split_on_space: bool) -> None:
        super().__init__()
        self._tokenizer_base = Tokenizer(BPE())
        self._tokenizer_base.add_special_tokens(SpecialTokens.values())
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        # use_regex=True uses the GPT2 regexp for spliting on whitespace
        # TODO: maybe change this?
        if split_on_space:
            self._tokenizer_base.normalizer = Sequence([NFD(), StripAccents(), Lowercase()])
        else:
            self._tokenizer_base.normalizer = Sequence([NFD(), StripAccents(), Lowercase(), Replace(" ", "")])
        self._tokenizer_base.pre_tokenizer = PTSequence([Digits(individual_digits=False),
                                                        ByteLevel(add_prefix_space=split_on_space, use_regex=split_on_space)])
        self._tokenizer_base.decoder = ByteLevelDecoder()
        self._tokenizer_base.post_processor = TemplateProcessing(
            single=f"{SpecialTokens.BOS} $A {SpecialTokens.EOS}",
            special_tokens=[
                (SpecialTokens.BOS, self._tokenizer_base.token_to_id(SpecialTokens.BOS)),
                (SpecialTokens.EOS, self._tokenizer_base.token_to_id(SpecialTokens.EOS)),
            ],
        )


    def tokenizer_base(self) -> Tokenizer:
        return self._tokenizer_base
    

    def train(self, dataset: list[str], initial_alphabet: list[str]):
        """Trains the tokenizer on the given dataset and initial alphabet."""
        trainer = BpeTrainer(vocab_size=self.vocab_size,
                             min_frequency=self.min_frequency,
                             show_progress=True,
                             special_tokens=SpecialTokens.values(),
                             initial_alphabet=initial_alphabet,
                             )
        self._tokenizer_base.train_from_iterator(dataset, trainer, length=len(dataset))

def get_desired_vocab_size(step: int, initial_alphabet: list[str]):
    """Gets the desired vocab size at this step.
    Minimally, the vocab size should be the same size as initial_alphabet."""
    # TODO: Make this more sensible
    return int(exp(step+1) + len(initial_alphabet))


def main(
        # Tokenizer Parameters
        tokenizer_type: str = "BPE",
        incremental: bool | None = False,
        retrain: bool = True,
        split_on_space: bool = True, # whether to split on space + punctuation, or not
        output_dir: Path = PROJECT_ROOT / "outputs",
        vocab_size: int = 30000,
        min_frequency: int | None = 15,
        bpe_batches: int | None = 10,
        # Data Parameters
        large_track: bool = False,
        subsample: int | None = None,
):
    # create project directory inside output_dir based on the timestamp
    # plus a two-character random string
    project_dir = (
        output_dir
        / f"{datetime.today().strftime('%Y-%m-%d-%H%M%S')}_{os.urandom(2).hex()}"
    )
    if not output_dir.exists():
        output_dir.mkdir()
    if not project_dir.exists():
        project_dir.mkdir()

    data = load_data(large_track=large_track, subsample=subsample)
    dataset = data["dataset"]["train"]
    all_chars = data["all_chars"]

    tokenizer_hps = {
        "tokenizer_type": tokenizer_type,
        "retrain": retrain,
        "split_on_space": split_on_space,
        "incremental": incremental,
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "bpe_batches": bpe_batches if incremental else 1,
        "large_track": large_track,
        "subsample": subsample,
    }

    print(f"Config: {pformat(tokenizer_hps)}")

    with open(project_dir / "tokenizer_hps.json", "w") as f:
        json.dump(tokenizer_hps, f)


    initial_alphabet = list(all_chars)
    previous_alphabet = initial_alphabet
    if incremental:
        step_size = int(len(dataset) / bpe_batches)
        end_points = range(0, len(dataset), step_size)
        if retrain:
            steps = [(0, step) for step in end_points]
            steps[-1] = ((0, len(dataset)))
        else:
            steps = list(pairwise(end_points))
            steps[-1] = (end_points[-2], len(dataset)) 
        desired_vocab_sizes = [get_desired_vocab_size(i, initial_alphabet) for i in range(len(steps))]
    else: # not incremental; just do one step
        steps = [(0,len(dataset))]
        desired_vocab_sizes = [vocab_size,]

    for i, ((s, e), desired_vocab_size) in enumerate(zip(steps, desired_vocab_sizes)):
        print(f"Start: {s}, End: {e}, Vocab Size: {desired_vocab_size}")
        trainer = BPETokenizerTrainer(vocab_size=desired_vocab_size, min_frequency=min_frequency, split_on_space=split_on_space)

        trainer.train(dataset=dataset["text"][s:e], initial_alphabet=previous_alphabet)
        trainer.tokenizer_base().save(str(project_dir/ f"tokenizer_{i}.json"))
        previous_alphabet = list(trainer.tokenizer_base().get_vocab().keys())

    # tokenizer = trainer.get_tokenizer()
    # encoded = tokenizer.encode("This is a full sentence with longlonglong words that are sesquepedalian.")
    # print(encoded)
    # print(tokenizer.convert_ids_to_tokens(encoded, skip_special_tokens=False))
    # print(tokenizer.decode(encoded))


if __name__ == "__main__":
    fire.Fire(main)