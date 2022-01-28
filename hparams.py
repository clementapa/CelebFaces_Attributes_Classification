from dataclasses import dataclass
import os, os.path as osp
from typing import Any, ClassVar, Dict, List, Optional

import simple_parsing
from simple_parsing.helpers import Serializable, choice, dict_field, list_field


@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    # wandb parameters
    wandb_project : str  = "classif_celeba"
    wandb_entity  : str  = "attributes_classification_celeba"       # name of the project
    save_dir      : str  = osp.join(os.getcwd())                    # directory to save wandb outputs
    weights_path  : str  = osp.join(os.getcwd(), "weights")

    # train or predict 
    train : bool = True
    predict: bool = False

    gpu: int = 0
    fast_dev_run: bool = False
    limit_train_batches: float = 1.0
    val_check_interval: float = 0.5

@dataclass
class TrainParams:
    """Parameters to use for the model"""
    model_name        : str         = "vit_small_patch16_224"
    pretrained        : bool        = True
    n_classes         : int         = 40 
    lr : int = 0.00001

@dataclass
class DatasetParams:
    """Parameters to use for the model"""
    # datamodule
    num_workers       : int         = 2         # number of workers for dataloadersint
    root_dataset      : Optional[str] = osp.join(os.getcwd(), "assets")   # '/kaggle/working'
    batch_size        : int         = 6        # batch_size
    input_size        : tuple       = (224, 224)   # image_size

@dataclass
class CallBackParams:
    """Parameters to use for the logging callbacks"""

    nb_image : int = 8
    early_stopping_params     : Dict[str, Any] = dict_field(
        dict(
            monitor="val/F1", 
            patience=10,
            mode="max",
            verbose=True
        )
    )
    model_checkpoint_params    : Dict[str, Any] = dict_field(
        dict(
            monitor="val/F1", 
            dirpath= osp.join(os.getcwd(), "weights"), #'/kaggle/working/', 
            filename="best-model",
            mode="max",
            verbose=True
        )
    )

@dataclass
class InferenceParams:
    """Parameters to use for the logging callbacks"""
    model_name        : str         = "vit_small_patch16_224"
    pretrained        : bool        = True
    n_classes         : int         = 40 
    ckpt_path: Optional[str] = None 

@dataclass
class Parameters:
    """base options."""

    hparams       : Hparams         = Hparams()
    data_param    : DatasetParams   = DatasetParams()
    callback_param: CallBackParams  = CallBackParams()
    train_param   : TrainParams     = TrainParams()
    inference_param : InferenceParams = InferenceParams()

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
