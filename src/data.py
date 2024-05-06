"""Construct BabyLM Dataset and initial tokenizer."""

import os
import re
import unicodedata
import zipfile
from functools import partial, reduce
from multiprocessing import Pool

import pyrootutils
import requests
import torch
from character_tokenizer import CharacterTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from osfclient import cli
from transformers import PreTrainedTokenizerFast
from unidecode import unidecode

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


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


def normalize_string(s: str, remove_spaces: bool) -> str:
    """Normalize a string."""

    def strip_accents(s: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c)
        )

    def lower(s: str) -> str:
        return s.lower()

    def collapse_spaces(s: str) -> str:
        return re.sub(r"\s+", " ", s)

    def removing_spaces(s: str) -> str:
        return re.sub(r"\s+", "", s)

    norm_maps = [strip_accents, lower, unidecode, collapse_spaces]
    if remove_spaces:
        norm_maps.append(removing_spaces)

    return reduce(lambda x, f: f(x), norm_maps, s)


def preprocess(
    examples: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    trunc: bool = True,
    max_len: int = 512,
    remove_spaces: bool = False,
) -> DatasetDict:
    """Tokenize dataset."""
    if trunc:
        return tokenizer(
            [
                normalize_string(x, remove_spaces=remove_spaces)
                for x in examples["text"]
            ],
            truncation=trunc,
            max_length=max_len,
        )
    else:
        return tokenizer(
            [
                normalize_string(x, remove_spaces=remove_spaces)
                for x in examples["text"]
            ],
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
    normalized_input = normalize_string(example["text"], remove_spaces=False)
    for char in normalized_input:
        codepoints.add(normalize_string(char, remove_spaces=False))
    return codepoints


def load_data(
    large_track: bool, subsample: int | None, seed: int, remove_spaces: bool
) -> dict:
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

    num_workers = os.cpu_count()
    pool = Pool(processes=num_workers)

    train_chars = pool.map(get_chars, dataset["train"])
    val_chars = pool.map(get_chars, dataset["validation"])
    test_chars = pool.map(get_chars, dataset["test"])
    all_chars = set.union(*train_chars, *val_chars, *test_chars)

    if remove_spaces and " " in all_chars:
        # Don't want to accidentally add-in spaces that will never be seen in
        # the processed data.
        all_chars.remove(" ")

    return {
        "dataset": dataset,
        "all_chars": all_chars,
    }


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
    merge_scores = (
        bigram_probs / unigram_probs
    ) * total_merge_counts  # P(wi|wi-1)/ P(wi)
    merge_scores = torch.nan_to_num(merge_scores)
    merge_scores *= prev_merged

    # zero out the merge_score for every row-column pair where either
    # index corresponds to a token that is not an alphabetic character.
    # This ensures that all valid pairs are ranked ahead of invalid pairs.
    current_specials = tokenizer.all_special_tokens
    current_vocab = list(tokenizer.get_sorted_vocab().keys())
    valid_to_merge = torch.tensor(
        [
            # With a CharacterTokenizer this second check is unnecessary,
            # but we include it in case the tokenizer is every changed to
            # allow fully-alphabet special tokens.
            x.isalpha() and x not in current_specials
            for x in current_vocab
        ],
        dtype=torch.bool,
        device=model.device,
    )
    merge_scores[~valid_to_merge, :] = 0
    merge_scores[:, ~valid_to_merge] = 0

    # get top num_to_merge valid token pairs
    merge_scores_ranked = merge_scores.flatten().argsort(descending=True)
    merge_scores_ranked = torch.stack(
        (
            merge_scores_ranked // merge_scores.shape[0],
            merge_scores_ranked % merge_scores.shape[0],
        )
    )
    top_alphas = merge_scores_ranked[:, :num_to_merge]

    # We filter again based on mergeability to ensure that we only every merge
    # the top min(num_to_merge, # of valid pairs) tokens, just in case we're
    # in a position where there are fewer than num_to_merge valid pairs.
    top_alphas = top_alphas[
        :,
        torch.logical_and(valid_to_merge[top_alphas[0]], valid_to_merge[top_alphas[1]]),
    ]

    new_toks = [
        tokenizer.decode(merge_scores_ranked[0, x])
        + tokenizer.decode(merge_scores_ranked[1, x])
        for x in range(top_alphas.shape[1])
    ]

    # update counters
    alpha_toks = torch.cat(
        (
            alpha_toks,
            torch.arange(
                len(tokenizer), len(tokenizer) + len(new_toks), device=model.device
            ),
        )
    )
    prev_merged[top_alphas] = False

    # add tokens to tokenizer
    tokenizer.add_tokens(new_toks)

    # update model's embedding matrix for new tokens as average of pair
    # if you know of a better way to do this; feel free to modify
    # could also initialize model to full target embedding size and fill in
    # new tokens rather than resizing each time.
    model.resize_token_embeddings(len(tokenizer))
    model_embs = model.get_input_embeddings()
    new_embs = model_embs.weight[top_alphas].mean(axis=0)
    model_embs.weight[len(tokenizer) - len(new_toks) :] = new_embs
    model.set_input_embeddings(model_embs)
    model.tie_weights()
    print("Newly added tokens", new_toks)
    return tokenizer, model, alpha_toks, prev_merged


def construct_dataset(
    seed: int,
    block_size: int,
    large_track: bool,
    subsample: int | None,
    stack: bool,
    tokenizer: PreTrainedTokenizerFast | None,
    remove_spaces: bool,
):
    """Construct BabyLM dataset and initial tokenizer."""
    # Check if PROJECT_ROOT / data has more than a single .gitkeep file in it
    if not len(list(PROJECT_ROOT.glob("data/*"))) > 1:
        download_data()

    data = load_data(
        large_track=large_track,
        subsample=subsample,
        seed=seed,
        remove_spaces=remove_spaces,
    )
    dataset = data["dataset"]
    all_chars = data["all_chars"]

    if tokenizer is None:
        tokenizer = CharacterTokenizer(
            all_chars,
            model_max_length=block_size,
            split_on_whitespace=not remove_spaces,
        )
    else:
        """
        If we already have a tokenzer, we want to construct an entirely new
        tokenizer with the union of the old vocab and the new characters from
        the dataset. However, we need to maintain the order of the old vocabulary
        since otherwise the model will be seeing randomized tokens at every epoch.

        To do this, we first strip out the special tokens from the old vocabulary.
        Then we take the current vocab and make sure that it's ordered according to
        it's token_id. We then remove these known tokens from the new characters,
        and concatenate current_vocab with new_vocab.
        """
        current_specials = tokenizer.all_special_tokens
        current_vocab = tokenizer.get_sorted_vocab()
        normal_vocab = {
            k: v for k, v in current_vocab.items() if k not in current_specials
        }
        new_chars = list(set(all_chars) - set(normal_vocab.keys()))
        new_chars = [k for k in normal_vocab] + new_chars
        tokenizer = CharacterTokenizer(
            new_chars,
            model_max_length=block_size,
            split_on_whitespace=not remove_spaces,
        )

    preprocess_fn = partial(
        preprocess,
        tokenizer=tokenizer,
        trunc=not stack,
        max_len=block_size,
        remove_spaces=remove_spaces,
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
