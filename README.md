# Attributes classification on CelebA dataset: Multi-label classification task

> Authors: [Apavou Clément](https://github.com/clementapa) & [Belkada Younes](https://github.com/younesbelkada)

![Python](https://img.shields.io/badge/Python-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-orange.svg?style=plastic)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-blueviolet.svg?style=plastic)

## :mag_right: Introduction :
This repository is related to a project of the Introduction to Numerical Imaging (i.e, Introduction à l'Imagerie Numérique in French), given by the MVA Masters program at ENS-Paris Saclay.

It was entirely build from scratch and contains code in PyTorch Lightning to train and then use a neural network for image classification. We used it to create a classifier allowing semantic attributes classification of faces with the dataset CelebA. 

<p align="center">
    <img src="assets/overview_celeba.png" width="500" height="300"/>
</p>
<p align="center">
<em> Some images of the CelebA dataset with attribute annotation.</em>
</p>

## :chart_with_upwards_trend: Experiments :

The dataset CelebA contains approximately 200,000 images of celebrities faces with 40 binary semantic attribute annotations such as smiling :grin: / not smiling :neutral_face: or bald :older_man: / not bald :man:. All attributes are available [here](https://github.com/clementapa/CelebFaces_Attributes_Classification/blob/master/utils/constant.py). 

We have fine-tuned two classifier a ResNet-50 and a ViT small with 16x16 patches. The training set contains 200,000 images, so only one epoch is sufficient to fine tune models to perform for attributes classification on CelebA dataset. 

Experiments are available on wandb: [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/classif_celeba?workspace=user-clementapa). 

## :mag_right: Results:

<p align="center">
    
| Model| Accuracy | Weights   | Run  |
|---|---|---|---|
| [vit_small_patch16_224](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py) | 0.7622  | [here](https://wandb.ai/attributes_classification_celeba/classif_celeba/artifacts/model/model-23z2z7bn/v5/files)  | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/classif_celeba/runs/23z2z7bn?workspace=user-clementapa) |
| [resnet50](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py) | 0.8055 | not available  | [![](https://github.com/wandb/assets/blob/main/wandb-github-badge-gradient.svg)](https://wandb.ai/attributes_classification_celeba/classif_celeba/runs/2xms83j2?workspace=user-clementapa)  | 
    
</p>

## :tada: Features :

For models, we used the [timm](https://fastai.github.io/timmdocs/) library providing many models for image classification. All image classification models from this library can be used. 

The entire code contains the following features :
- Training of a neural network for image classification for any dataset (you just have to add your custom dataset in the folder datasets) 
- Visualisation with the library Weights and Biases of several metrics of classification such as losses, accuracy, précision and recall. Also, prédictions of some training and validation images are logged on wandb to follow in real time the progress of the training. 
- Inference of the model on your own images. You just have to add your images whose you want the model to infer and specify the path of the folder in the config file 

The first goal of this repository was to use the InterFaceGAN method. So, there is a script train_svm.py which allows to train SVM for each semantic attributes to obtain boundaries and use them to control faces generation with InterfaceGAN of [this repository](https://github.com/younesbelkada/interfacegan). 

The all code is useable by just modifying the config file (config/hparams.py). You can launch a training of classifier, launch an inférence of a classifier (by using weigths of à trained classifier) and you can train SVMs to create boundaries. 

## :dart: Code structure :
The structure of repository is the following :

```
├── assets                      # Put database here
├── datamodules
|   |
|   ├── celebadatamodule.py     # datamodules PyTorch lightning for CelebA dataset
|         
├── datasets
|   ├── celeba.py                # Fix issue for CelebA dataset PyTorch
|   ├── inference_dataset.py     # custom dataset PyTorch for inference
|          
├── lightningmodules
|   ├── classification.py        # lightning module for image classification (multi-label)
| 
├── utils                        # utils functions
|   ├── boundary_creator.py
|   ├── callbacks.py
|   ├── constant.py
|   ├── utils_functions.py
|
├── weights                     # put models weights here
|
├── analyse_score_latent_space.ipynb  # notebook to analyse scores predicted
|
├── hparams.py                   # configuration file
|
├── main.py                      # main script to launch for training of inference 
| 
├── train_svm.py                 # script to create boundaries for InterFaceGAN
|
└── README.md
```

## :hammer: Usage :

### Train a classifier

Parameters to put in ```hparams.py```:
```
    train : bool = True
    predict: bool = False 
```

Then change ```Hparams```, ```TrainParams```, ```DatasetParams``` and ```CallBackParams``` with your needs.

```
python main.py
```

### Predict with the classifier

Parameters to put in ```hparams.py```:
```
    train : bool = False
    predict: bool = True 
```

Then change ```Hparams```, ```InferenceParams``` and ```DatasetParams``` with your needs.

```
python main.py
```

### Train SVM for InterFaceGAN

Modify ```SVMParams``` in ```hparams.py``` with your needs.

```
python train_svm.py
```
