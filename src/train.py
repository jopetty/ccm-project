"""Train models."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from pprint import pformat
from random import randint

import fire
import flax
import jax
import jax.numpy as jnp
import optax
import pyrootutils
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dotenv import load_dotenv
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from tqdm import tqdm
from transformers import (
    FlaxAutoModelForCausalLM,
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


def data_loader(rng, dataset, batch_size, shuffle=False):
    """Flax data loader."""
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}

        batch = shard(batch)

        yield batch


def train_step(state, batch, dropout_rng, schedule_fn):
    """Flax training step."""
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        labels = batch.pop("labels")
        logits = state.apply_fn(
            **batch, params=params, dropout_rng=dropout_rng, train=True
        )[0]

        loss = optax.softmax_cross_entropy(
            logits[..., :-1, :], onehot(labels[..., 1:], logits.shape[-1])
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.lax.pmean(grad, "batch")
    new_state = state.apply_gradients(grads=grad)

    metrics = jax.lax.pmean(
        {"loss": loss, "learning_rate": schedule_fn(state.step)}, axis_name="batch"
    )

    return new_state, metrics, new_dropout_rng


def eval_step(params, batch, model):
    """Flax evaluation step."""
    labels = batch.pop("labels")

    logits = model(**batch, params=params, train=False)[0]

    loss = optax.softmax_cross_entropy(
        logits[..., :-1, :], onehot(labels[..., 1:], logits.shape[-1])
    ).mean()

    # summarize metrics
    metrics = {"loss": loss, "perplexity": jnp.exp(loss)}
    metrics = jax.lax.pmean(metrics, axis_name="batch")
    return metrics


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
    weight_decay: float = 0.01,
    eval_every: int = 1,
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
):
    """Train models."""
    set_seed(seed)
    os.environ["WANDB_PROJECT"] = project_name

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

    print(f"Config: {pformat(project_hps)}")

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
    model = FlaxAutoModelForCausalLM.from_config(
        gpt_config, seed=seed, dtype=jnp.dtype("bfloat16")
    )

    print(f"Model: {model}")

    # Training
    total_batch_size = per_device_batch_size * jax.device_count()
    num_train_steps = len(dataset["train"]) // total_batch_size * num_epochs

    linear_decay_lr_schedule_fn = optax.linear_schedule(
        init_value=lr, end_value=0, transition_steps=num_train_steps
    )
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=beta1,
        b2=beta2,
        eps=1e-8,
        weight_decay=weight_decay,
    )

    state = train_state.TrainState.create(
        apply_fn=model.__call__, params=model.params, tx=adamw
    )

    parallel_train_step = jax.pmap(train_step, "batch")
    parallel_eval_step = jax.pmap(eval_step, "batch")

    state = flax.jax_utils.replicate(state)

    rng = jax.random.PRNGKey(seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    for epoch in tqdm(
        range(1, num_epochs + 1), desc="Epoch ...", position=0, leave=True
    ):
        rng, input_rng = jax.random.split(rng)

        train_loader = data_loader(
            input_rng, dataset["train"], total_batch_size, shuffle=True
        )
        with tqdm(
            total=len(dataset["train"]) // total_batch_size,
            desc="Training...",
            leave=False,
        ) as pbar_train:
            for model_inputs in train_loader:
                state, train_metric, dropout_rngs = parallel_train_step(
                    state, model_inputs, dropout_rngs, linear_decay_lr_schedule_fn
                )
                pbar_train.update(1)
            pbar_train.write(
                f"Train... ({epoch}/{num_epochs} | Loss: {round(train_metric['loss'].mean(), 3)}, Learning Rate: {round(train_metric['learning_rate'].mean(), 6)})"  # noqa: E501
            )

        eval_loader = data_loader(input_rng, dataset["validation"], total_batch_size)
        eval_metrics = []
        with tqdm(
            total=len(dataset["validation"]) // total_batch_size,
            desc="Evaluation...",
            leave=False,
        ) as pbar_eval:
            for model_inputs in eval_loader:
                eval_metric = parallel_eval_step(state.params, model_inputs)
                eval_metrics.append(eval_metric)
                pbar_eval.update(1)

            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.mean, eval_metrics)
            pbar_eval.write(
                f"Eval... ({epoch}/{num_epochs} | Loss: {eval_metrics['loss']} | Perplexity: {eval_metrics['perplexity']})"  # noqa: E501
            )

    model.save_pretrained(project_dir)


if __name__ == "__main__":
    fire.Fire(main)
