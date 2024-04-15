"""Custom W&B Metrics & Callbacks."""

from transformers.integrations import WandbCallback


class ExtendConfigCallback(WandbCallback):
    """Callback to update wandb config with custom hyperparameters."""

    def __init__(self, hps_dict):  # noqa: D107
        super().__init__()
        self.hps_dict = hps_dict

    def setup(self, args, state, model, **kwargs):  # noqa: D102
        super().setup(args, state, model, **kwargs)
        self._wandb.config.update(self.hps_dict)
