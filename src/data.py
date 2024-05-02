"""Construct BabyLM Dataset and initial tokenizer."""

import os
import zipfile
from enum import StrEnum
from functools import partial
from multiprocessing import Pool

import pyrootutils
from datasets import Dataset, DatasetDict, load_dataset
from osfclient import cli
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


class SpecialTokens(StrEnum):
    """Special tokens for tokenizer."""

    BOS = "[BOS]"
    UNK = "[UNK]"
    EOS = "[EOS]"
    SEP = "[SEP]"
    CLS = "[CLS]"
    MASK = "[MASK]"

    @classmethod
    def values(cls):
        """Return a list of the string values of each special token."""
        return list(map(lambda c: c.value, cls))

    @property
    def index(self):
        """Return the index of the token in the vocabulary.

        Used to get the index of the PAD token when directly modifying tensors.
        """
        return SpecialTokens.values().index(self.value)


class OSFArgs:
    """Args for the OSF client."""

    project: str
    remote: str | None = None
    local: str | None = None
    username: str | None = None
    force: bool = True
    update: bool = True

    def __init__(  # noqa: D107
        self, project: str, remote: str | None = None, local: str | None = None
    ):
        self.project = project
        self.remote = remote
        self.local = local
        self.username = None
        self.force = True
        self.update = True


def tokenize(examples: DatasetDict, tokenizer: PreTrainedTokenizerFast) -> DatasetDict:
    """Tokenize dataset."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )


def stack_sequences(examples: DatasetDict, block_size: int | None = None):
    """Sequence stacking."""
    if block_size is None:
        examples["labels"] = examples["input_ids"].copy()
        return examples

    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()

    return result


def download_data():
    """Download & unzip BabyLM data."""
    baby_lm_files = ["dev.zip", "test.zip", "train_100M.zip", "train_10M.zip"]

    for file in baby_lm_files:
        args = OSFArgs(
            project="ad7qg",
            remote=f"text_data/{file}",
            local=str(PROJECT_ROOT / "data" / file),
        )
        print("Downloading", args.remote, "to", args.local)
        cli.fetch(args)

    for file in baby_lm_files:
        zip_path = PROJECT_ROOT / "data" / file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(PROJECT_ROOT / "data")
            print("Extracted", file)


def get_chars(example: Dataset) -> set:
    """Get all characters in dataset."""
    codepoints = set()
    text = example["text"]
    for char in text:
        codepoints.add(char)
    return codepoints


def load_data(large_track: bool, subsample: int | None) -> dict:
    """Load BabyLM data into HF dataset object."""
    if large_track:
        data_files = {
            "train": str(PROJECT_ROOT / "data" / "train_100M/*.train"),
            "validation": str(PROJECT_ROOT / "data" / "dev/*.dev"),
            "test": str(PROJECT_ROOT / "data" / "test/*.test"),
        }
    else:
        data_files = {
            "train": str(PROJECT_ROOT / "data" / "train_10M/*.train"),
            "validation": str(PROJECT_ROOT / "data" / "dev/*.dev"),
            "test": str(PROJECT_ROOT / "data" / "test/*.test"),
        }

    dataset = load_dataset("text", data_files=data_files)

    if subsample is not None:
        dataset["train"] = dataset["train"].select(range(subsample))
        dataset["validation"] = dataset["validation"].select(range(subsample))
        dataset["test"] = dataset["test"].select(range(subsample))

    print(dataset)

    num_workers = 8
    pool = Pool(processes=num_workers)

    train_chars = pool.map(get_chars, dataset["train"])
    val_chars = pool.map(get_chars, dataset["validation"])
    test_chars = pool.map(get_chars, dataset["test"])
    all_chars = set.union(*train_chars, *val_chars, *test_chars)

    return {
        "dataset": dataset,
        "all_chars": all_chars,
    }


def get_initial_tokenizer(unique_tokens: set[str]):
    """Get char-level tokenizer."""
    tokenizer_base = Tokenizer(WordLevel())
    tokenizer_base.pre_tokenizer = WhitespaceSplit()
    tokenizer_base.add_special_tokens(SpecialTokens.values())
    tokenizer_base.add_tokens(list(unique_tokens))

    tokenizer_base.post_processor = TemplateProcessing(
        single=f"{SpecialTokens.BOS} $A {SpecialTokens.EOS}",
        special_tokens=[
            (SpecialTokens.BOS, tokenizer_base.token_to_id(SpecialTokens.BOS)),
            (SpecialTokens.EOS, tokenizer_base.token_to_id(SpecialTokens.EOS)),
        ],
    )

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_base,
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


def construct_dataset(
    seed: int, block_size: int, large_track: bool, subsample: int | None
):
    """Construct BabyLM dataset and initial tokenizer."""
    # Check if PROJECT_ROOT / data has more than a single .gitkeep file in it
    if not len(list(PROJECT_ROOT.glob("data/*"))) > 1:
        download_data()

    data = load_data(large_track=large_track, subsample=subsample)
    dataset = data["dataset"]
    all_chars = data["all_chars"]

    tokenizer = get_initial_tokenizer(all_chars)

    tokenize_map = partial(tokenize, tokenizer=tokenizer)
    dataset = dataset.map(
        tokenize_map,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=dataset["train"].column_names,
    )

    # Shuffle before stacking
    dataset["train"] = dataset["train"].shuffle(seed=seed)

    stack_map = partial(stack_sequences, block_size=block_size)
    dataset = dataset.map(
        stack_map,
        batched=True,
        num_proc=os.cpu_count(),
    )

    return {
        "dataset": dataset,
        "tokenizer": tokenizer,
    }


if __name__ == "__main__":
    load_data()
