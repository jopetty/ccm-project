"""Train models."""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from random import randint, sample

import fire
import pyrootutils
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    GPT2Config,
    Trainer,
    TrainingArguments,
)

from data import construct_dataset, merge_new_tokens

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)
log = get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


load_dotenv()


def main(
    # Model Parameters
    d_model: int = 512,
    n_heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
    activation: str = "gelu",
    layer_norm_eps: float = 1e-5,
    norm_first: bool = True,
    n_layers: int = 6,
    # Training Parameters
    num_epochs: int = 1,
    per_device_batch_size: int = 16,
    lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    compile: bool = False,
    weight_decay: float = 0.01,
    eval_every: int = 100,
    gradient_clip: float | None = 1.0,
    block_size: int = 512,
    num_vocab_merges_per_step: int = 50,
    update_vocab_every: int = 100,
    update_vocab: bool = True,
    # Dataset Parameters
    large_track: bool = False,
    subsample: int | None = None,
    stack_sequences: bool = True,
    # Miscellaneous
    output_dir: Path = PROJECT_ROOT / "outputs",
    seed: int = randint(0, 2**32 - 1),
    project_name: str = "ccm_project_test",
    logging: bool = True,
):
    """Train models."""
    set_seed(seed)

    # accelerator = Accelerator(log_with="wandb") if logging else Accelerator()
    accelerator = Accelerator()

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

    dataset_dict = construct_dataset(
        large_track=large_track,
        seed=seed,
        subsample=subsample,
        block_size=block_size,
        stack=stack_sequences,
        tokenizer=None,
    )

    dataset = dataset_dict["dataset"]
    tokenizer = dataset_dict["tokenizer"]

    print(f"Dataset: {dataset}")

    tokenizer.save_pretrained(project_dir)

    project_hps = {
        "activation": activation,
        "batch_first": True,
        "per_device_batch_size": per_device_batch_size,
        "betas": (beta1, beta2),
        "block_size": block_size,
        "compile": compile,
        "d_model": d_model,
        "d_ff": d_ff,
        "dropout": dropout,
        "epochs": num_epochs,
        "eval_every": eval_every,
        "gradient_clip": gradient_clip,
        "layer_norm_eps": layer_norm_eps,
        "lr": lr,
        "large_track": large_track,
        "n_heads": n_heads,
        "norm_first": norm_first,
        "n_layers": n_layers,
        "project_dir": str(project_dir),
        "seed": seed,
        "subsample": subsample,
        "stack_sequences": stack_sequences,
        "weight_decay": weight_decay,
        "update_vocab_every": update_vocab_every,
        "num_vocab_merges_per_step": num_vocab_merges_per_step,
    }

    accelerator.init_trackers(
        project_name,
        config=project_hps,
    )

    print(f"Config: {project_hps}")
    # log.info(f"Dataset: {dataset}")

    with open(project_dir / "project_hps.json", "w") as f:
        json.dump(project_hps, f)

    gpt_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=block_size,
        n_embd=d_model,
        n_layer=n_layers,
        n_head=n_heads,
        n_inner=d_ff,
        activation_function=activation,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    model = AutoModelForCausalLM.from_config(
        gpt_config,
    )

    log.info(f"Accelerator state: {accelerator.state}")

    training_args = TrainingArguments(
        output_dir=project_dir,
        logging_strategy="epoch",
        logging_first_step=True,
        learning_rate=lr,
        load_best_model_at_end=True,
        weight_decay=weight_decay,
        push_to_hub=False,
        max_grad_norm=gradient_clip,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        torch_compile=False,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        save_total_limit=1,
        save_safetensors=True,
        save_strategy="no",
        seed=seed,
        report_to="none",
    )

    total_merge_probs = torch.zeros((len(tokenizer), len(tokenizer)))
    total_merge_counts = torch.zeros((len(tokenizer), len(tokenizer)))
    alpha_toks = set(
        [idx for tok, idx in tokenizer.get_added_vocab().items() if tok.isalpha()]
    )
    prev_merged = set()

    data_seeds = sample(range(1, 100), num_epochs)

    for e in range(num_epochs):
        print("#####################################")
        print(f"Epoch: {e}")
        print(f"{len(tokenizer)} tokens in tokenizer")
        print("#####################################")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
        )
        trainer.train()

        device = trainer.model.device

        total_merge_probs = total_merge_probs.to(device)
        total_merge_counts = total_merge_counts.to(device)

        vocab_dataloader = DataLoader(
            dataset["train"],
            shuffle=False,
            batch_size=per_device_batch_size,
            collate_fn=data_collator,
        )
        vocab_dataloader = accelerator.prepare(vocab_dataloader)

        model.eval()
        with torch.no_grad():
            start = time.time()
            for idx, v_batch in enumerate(vocab_dataloader):
                output = model(**v_batch)
                target = v_batch["labels"]
                logits = output.logits

                if idx == 0:
                    first_target = target[0]
                    first_target[first_target == -100] = tokenizer.eos_token_id
                    first_logits = logits[0].argmax(dim=-1)
                    print("Vocab batch:")
                    print(f"Target: {tokenizer.decode(
                        first_target, skip_special_tokens=not stack_sequences)}")
                    print(f"Prediction: {tokenizer.decode(
                        first_logits, skip_special_tokens=not stack_sequences)}")

                if update_vocab:
                    target = target.flatten()
                    logits = logits.flatten(end_dim=-2)
                    loss = F.cross_entropy(logits, target, reduction="none")

                    # update bigram prob and count tracker
                    # (had to do this iteratively bc memory overhead was too large if
                    # tensorized; hope its not too slow)
                    target_toks, probs = target[target >= 0], (-loss[target >= 0]).exp()

                    for w_i in range(1, target_toks.shape[0]):
                        total_merge_probs[target_toks[w_i - 1], target_toks[w_i]] += (
                            probs[w_i]
                        )
                        total_merge_counts[target_toks[w_i - 1], target_toks[w_i]] += 1

            end = time.time()
            print(f"Time to evaluate: {end - start}")

            if update_vocab:
                start = time.time()
                tokenizer, model, alpha_toks = merge_new_tokens(
                    total_merge_probs,
                    total_merge_counts,
                    num_vocab_merges_per_step,
                    tokenizer,
                    model,
                    alpha_toks,
                    prev_merged,
                )
                end = time.time()
                print(f"Time to merge tokens: {end - start}")
                dataset_dict = construct_dataset(
                    large_track=large_track,
                    seed=data_seeds[e],
                    subsample=subsample,
                    block_size=block_size,
                    tokenizer=tokenizer,
                    stack=stack_sequences,
                )
                dataset = dataset_dict["dataset"]

                # Pad total_merge_probs, total_merge_counts to match new tokenizer size
                new_len = len(tokenizer)
                old_len = total_merge_probs.shape[0]

                total_merge_probs = F.pad(
                    total_merge_probs, (0, new_len - old_len, 0, new_len - old_len)
                )
                total_merge_counts = F.pad(
                    total_merge_counts, (0, new_len - old_len, 0, new_len - old_len)
                )

                # prompt = "Hello, "
                # p_input = tokenizer(prompt, return_tensors="pt")
                # p_input = {k: v[:-1].to(device) for k, v in p_input.items()}
                # p_output = model.generate(
                #     **p_input,
                #     do_sample=True,
                #     num_beams=1,
                #     max_new_tokens=100,
                #     num_return_sequences=1,
                # )
                # print(tokenizer.decode(p_output[0], skip_special_tokens=False))

                tokenizer.save_pretrained(project_dir, filename_prefix=f"{e}")

    model.save_pretrained(project_dir)


if __name__ == "__main__":
    fire.Fire(main)
