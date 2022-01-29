import json
import os.path as osp
from datetime import datetime
from attr import attr

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichProgressBar)
from pytorch_lightning.loggers import WandbLogger

from datamodules.celebadatamodule import CelebADataModule
from hparams import Parameters
from lightningmodules.classification import Classification
from utils.callbacks import MetricsCallback, WandbImageCallback
from utils.constant import ATTRIBUTES
from utils.utils_functions import create_dir


def main():
    config = Parameters.parse()

    dataset_module = CelebADataModule(config.data_param, train=config.hparams.train)

    if config.hparams.train:

        model = Classification(config.train_param, ATTRIBUTES)

        wdb_config = {}
        for k, v in vars(config).items():
            for key, value in vars(v).items():
                wdb_config[f"{k}-{key}"] = value

        wandb_logger = WandbLogger(
            config=wdb_config,
            project=config.hparams.wandb_project,
            entity=config.hparams.wandb_entity,
            allow_val_change=True,
            save_dir=config.hparams.save_dir,
        )

        callbacks = [EarlyStopping(**config.callback_param.early_stopping_params),
                     MetricsCallback(config.train_param.n_classes),
                     WandbImageCallback(config.callback_param.nb_image),
                     ModelCheckpoint(**config.callback_param.model_checkpoint_params),
                     RichProgressBar(),
                     LearningRateMonitor(),
        ]

        trainer = Trainer(logger=wandb_logger,
                          gpus=config.hparams.gpu,
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
        output_dict = {"filenames":[], "logits":[], "converted_preds":[], "preds_with_conf":[]}
        model = Classification(config.inference_param, dataset_module.attr_dict)
        trainer = Trainer()
        predictions = trainer.predict(model, dataset_module, ckpt_path=config.inference_param.ckpt_path)
        
        output_root = config.inference_param.output_root
        create_dir(output_root)
        name_output = f"output_dict-{datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}.json"
        output_full_path = osp.join(output_root, name_output)

        for pred_batch in predictions:
            img_names, preds, converted_preds, converted_logits = pred_batch[0], pred_batch[1], pred_batch[2], pred_batch[3]
            # {"filenames":[], "logits":[], "converted_preds":[] }
            for i, img_name in enumerate(img_names):
                output_dict['filenames'].append(img_name)
                output_dict['logits'].append(converted_logits.tolist()[i])
                output_dict['converted_preds'].append(converted_preds[i])
                preds_with_conf = {ATTRIBUTES[idx]:round(converted_logits.tolist()[i][idx], 3) for idx in np.where(preds[i]==1.0)[0]}
                output_dict['preds_with_conf'].append(preds_with_conf)
        json.dump(output_dict, open(output_full_path, 'w'))

if __name__ == "__main__":
    main()
