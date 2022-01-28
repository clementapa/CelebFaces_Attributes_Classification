from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datamodules.celebadatamodule import CelebADataModule
from hparams import Parameters
from lightningmodules.classification import Classification
from utils.callbacks import MetricsCallback, WandbImageCallback


def main():
    config = Parameters.parse()

    dataset_module = CelebADataModule(config.data_param)

    if config.hparams.train:

        model = Classification(config.train_param, dataset_module.attr_dict)

        wdb_config = {}
        for k, v in vars(config).items():
            for key, value in vars(v).items():
                wdb_config[f"{k}-{key}"] = value

        wandb_logger = WandbLogger(
            config=wdb_config,
            project=config.wandb_project,
            entity=config.wandb_entity,
            allow_val_change=True,
            save_dir=config.save_dir,
        )

        callbacks = [EarlyStopping(config.callback_param.early_stopping_params),
                     MetricsCallback(config.train_param.n_classes),
                     WandbImageCallback(config.callback_param.nb_image),
                     ModelCheckpoint(config.callback_param.model_checkpoint_params)
        ]

        trainer = Trainer(logger=wandb_logger,
                          gpus=config.hparams.gpu,
                          auto_select_gpus=True,
                          # auto_scale_batch_size="power",
                          callbacks=callbacks,
                          log_every_n_steps=1,
                          enable_checkpointing=True,
                          fast_dev_run=config.hparams.fast_dev_run,
                          limit_train_batches=config.hparams.limit_train_batches,
                          val_check_interval=config.hparams.val_check_interval,
                          )

        trainer.fit(model, dataset_module)

    if config.hparams.predict:
        model = Classification(config.inference_param, dataset_module.attr_dict)

        trainer.predict(model, dataset_module, ckpt_path=config.ckpt_path)
