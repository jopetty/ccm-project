"""Train models."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from pprint import pformat
from random import randint

import fire
import humanize
import pyrootutils
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dotenv import load_dotenv
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    GPT2Config,
)

from data import construct_dataset

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
    stack_sequences: bool = True,
    # Dataset Parameters
    large_track: bool = False,
    subsample: int | None = None,
    # Miscellaneous
    output_dir: Path = PROJECT_ROOT / "outputs",
    seed: int = randint(0, 2**32 - 1),
    project_name: str = "ccm_project_test",
    logging: bool = True,
):
    """Train models."""
    set_seed(seed)

    accelerator = Accelerator(log_with="wandb") if logging else Accelerator()

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
        large_track=large_track, seed=seed, subsample=subsample, block_size=block_size
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
    }

    accelerator.init_trackers(
        project_name,
        config=project_hps,
    )

    log.info(f"Config: {pformat(project_hps)}")
    log.info(f"Dataset: {dataset}")

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

    log.info(f"Model: {model}")
    log.info(
        f"Number of parameters: {humanize.intword(model.num_parameters)}"
        f" ({model.num_parameters})"
    )
    log.info(f"Accelerator state: {accelerator.state}")

    device = accelerator.device

    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(
        dataset["train"],
        shuffle=False,
        batch_size=per_device_batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        dataset["validation"],
        shuffle=False,
        batch_size=per_device_batch_size,
        collate_fn=data_collator,
    )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    global_step = 0
    for _ in (n_bar := tqdm(range(num_epochs), desc="Epochs", position=0, leave=False)):
        model.train()
        for batch in tqdm(train_dataloader, desc="Train", position=1, leave=False):
            global_step += 1
            optimizer.zero_grad()

            output = model(**batch)

            target = batch["labels"].flatten()
            logits = output.logits.flatten(end_dim=-2)
            loss = F.cross_entropy(logits, target).item()

            accelerator.backward(loss)

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip, norm_type=2.0
                )

            optimizer.step()

            n_bar.set_postfix({"loss": f"{loss:.5f}"})
            accelerator.log(
                {
                    "train/custom_loss": loss,
                    "train/hf_loss": output.loss,
                },
                step=global_step,
            )

            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for batch in tqdm(
                        eval_dataloader, desc="Eval", position=1, leave=False
                    ):
                        output = model(**batch)
                        target = batch["labels"].flatten()
                        logits = output.logits.flatten(end_dim=-2)
                        loss = F.cross_entropy(logits, target)
                        val_losses.append((loss, output.loss))

                    eval_samples = len(val_losses)
                    custom_val_loss = sum([x[0] for x in val_losses]) / eval_samples
                    hf_val_loss = sum([x[1] for x in val_losses]) / eval_samples
                    accelerator.log(
                        {
                            "eval/custom_loss": custom_val_loss,
                            "eval/hf_loss": hf_val_loss,
                        },
                        step=global_step,
                    )
                model.train()

    accelerator.end_training()
    model.save_pretrained(project_dir)


if __name__ == "__main__":
    fire.Fire(main)
