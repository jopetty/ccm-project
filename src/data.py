"""Construct BabyLM Dataset and initial tokenizer."""

import os
import zipfile
from enum import StrEnum
from functools import partial
from multiprocessing import Pool

import pyrootutils
import requests
import torch
from datasets import Dataset, DatasetDict, load_dataset
from osfclient import cli
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from unidecode import unidecode

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
    PAD = "[PAD]"
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


def preprocess(
    examples: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    trunc: bool = True,
    max_len: int = 512,
) -> DatasetDict:
    """Tokenize dataset."""
    if trunc:
        return tokenizer(
            [unidecode(x).lower() for x in examples["text"]],
            truncation=trunc,
            max_length=512,
        )
    else:
        return tokenizer(
            [unidecode(x).lower() for x in examples["text"]],
        )


def stack_sequences(examples: DatasetDict, block_size: int):
    """Sequence stacking."""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def load_references():
    """Download and format reference data."""
    reference_files = {
        "wordlist.txt": "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
        "sigmorphon_train.tsv": "https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/eng.word.train.tsv",
        "sigmorphon_dev.tsv": "https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/eng.word.dev.tsv",
        "aoa_ws_fit.csv": "https://gist.githubusercontent.com/craaaa/1c254cdc29bbe3f9ab25d66afc3ecfa3/raw/079968f76b94a1d38da066306c5b8688d2927018/gistfile1.txt",
    }
    os.makedirs(PROJECT_ROOT / "data/references", exist_ok=True)

    for fname, url in reference_files.items():
        print(f"Getting {fname}")
        get_response = requests.get(url)
        if get_response.ok:
            with open(PROJECT_ROOT / "data/references" / fname, "w") as f:
                f.write(get_response.text)
        else:
            print(f"Could not obtain {fname} from {url}")

    morphemes = set()
    files = [
        PROJECT_ROOT / "data/references" / "sigmorphon_train.tsv",
        PROJECT_ROOT / "data/references" / "sigmorphon_dev.tsv",
    ]
    for fname in files:
        with open(fname, "r") as f:
            for line in f:
                word, morphs, _ = line.split("\t")
                m = morphs.strip().replace("@@", "").split(" ")
                morphemes.update(m)

    with open(PROJECT_ROOT / "data/references" / "sigmorphon_morphemes.txt", "w") as f:
        for line in morphemes:
            f.write(line + "\n")


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
        codepoints.add(unidecode(char).lower())
    return codepoints


def load_data(large_track: bool, subsample: int | None, seed: int) -> dict:
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
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(subsample))
        dataset["validation"] = (
            dataset["validation"].shuffle(seed=seed).select(range(subsample))
        )
        dataset["test"] = dataset["test"].shuffle(seed=seed).select(range(subsample))

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
    tokenizer_base = Tokenizer(WordLevel(unk_token=SpecialTokens.UNK))
    # tokenizer_base.normalizer = normalizers.Sequence([
    #     NFD(),
    #     StripAccents(),
    #     Lowercase()
    # ])

    tokenizer_base.pre_tokenizer = WhitespaceSplit()
    tokenizer_base.add_special_tokens(SpecialTokens.values())
    tokenizer_base.add_tokens(list(unique_tokens))

    tokenizer_base.post_processor = TemplateProcessing(
        single=f"{SpecialTokens.BOS} $A",
        special_tokens=[
            (SpecialTokens.BOS, tokenizer_base.token_to_id(SpecialTokens.BOS)),
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
    tokenizer.add_special_tokens(
        {
            "pad_token": SpecialTokens.EOS,
            "mask_token": SpecialTokens.MASK,
            "sep_token": SpecialTokens.SEP,
            "cls_token": SpecialTokens.CLS,
            "unk_token": SpecialTokens.UNK,
            "bos_token": SpecialTokens.BOS,
            "eos_token": SpecialTokens.EOS,
        }
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def merge_new_tokens(
    total_merge_probs,
    total_merge_counts,
    num_to_merge,
    tokenizer,
    model,
    alpha_toks,
    prev_merged,
):
    """Merges tokens based on merge scores."""
    # compute merge score - ratio of bigram conditional to
    # unigram probability of a token
    bigram_probs = total_merge_probs / total_merge_counts
    unigram_probs = total_merge_probs.sum(axis=0) / total_merge_counts.sum(axis=0)
    merge_scores = bigram_probs / unigram_probs  # P(wi|wi-1)/ P(wi)
    merge_scores = torch.nan_to_num(merge_scores)

    # get top num_to_merge valid token pairs
    merge_scores_ranked = merge_scores.flatten().argsort(descending=True)
    merge_scores_ranked = (
        merge_scores_ranked // merge_scores.shape[0],
        merge_scores_ranked % merge_scores.shape[0],
    )
    top_alphas = [
        (merge_scores_ranked[0][x].item(), merge_scores_ranked[1][x].item())
        for x in range(len(merge_scores_ranked[0]))
        if (merge_scores_ranked[0][x].item() in alpha_toks)
        and (merge_scores_ranked[1][x].item() in alpha_toks)
        and (
            (merge_scores_ranked[0][x].item(), merge_scores_ranked[1][x].item())
            not in prev_merged
        )
    ][:num_to_merge]
    new_toks = [tokenizer.decode(x) + tokenizer.decode(y) for x, y in top_alphas]

    # update counters
    alpha_toks.update(range(len(tokenizer), len(tokenizer) + len(new_toks)))
    prev_merged.update(top_alphas)

    # add tokens to tokenizer
    tokenizer.add_tokens(new_toks)

    # update model's embedding matrix for new tokens as average of pair
    # if you know of a better way to do this; feel free to modify
    # could also initialize model to full target embedding size and fill in
    # new tokens rather than resizing each time.
    model.resize_token_embeddings(len(tokenizer))
    model_embs = model.get_input_embeddings()
    new_embs = model_embs.weight[top_alphas].mean(axis=1)
    model_embs.weight[len(tokenizer) - len(new_toks) :] = new_embs
    model.set_input_embeddings(model_embs)
    model.tie_weights()
    print("Newly added tokens", new_toks)
    return tokenizer, model, alpha_toks


def construct_dataset(
    seed: int,
    block_size: int,
    large_track: bool,
    subsample: int | None,
    stack: bool,
    tokenizer: PreTrainedTokenizerFast | None,
):
    """Construct BabyLM dataset and initial tokenizer."""
    # Check if PROJECT_ROOT / data has more than a single .gitkeep file in it
    if not len(list(PROJECT_ROOT.glob("data/*"))) > 1:
        download_data()

    data = load_data(large_track=large_track, subsample=subsample, seed=seed)
    dataset = data["dataset"]
    all_chars = data["all_chars"]

    if tokenizer is None:
        tokenizer = get_initial_tokenizer(all_chars)
    else:
        tokenizer.add_tokens(list(all_chars))

    # print(tokenizer)

    preprocess_fn = partial(
        preprocess, tokenizer=tokenizer, trunc=not stack, max_len=block_size
    )
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=dataset["train"].column_names,
    )

    if stack:
        stack_fn = partial(stack_sequences, block_size=block_size)
        dataset = dataset.map(
            stack_fn,
            batched=True,
            num_proc=os.cpu_count(),
        )

    return {
        "dataset": dataset,
        "tokenizer": tokenizer,
    }


if __name__ == "__main__":
    load_references()
    load_data()
