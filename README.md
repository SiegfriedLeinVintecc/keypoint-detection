
<h1 align="center">Pytorch Keypoint Detection</h1>

This repo is a fork from the [keypoint detection repository by Thomas Lips](https://github.com/tlpss/keypoint-detection). I used this to try the different pre-made backbones using different hyperparameters on the dataset by Vintecc. I further added backbones and changed code were needed.
In this readme a guide to the code structure, models and use will be given.
To install and for other information, I leave the readme from Thomas' repository below.

## Code structure
- ```/keypoint_detection```: The main chunck of the code is found in this folder.

  - ```/keypoint_detection/data```: contains the code allowing the trainer to load, augment and use (COCO) datasets.
  
  - ```/keypoint_detection/models```:
    - ```/keypoint_detection/models/backones```: contains the backbones and a backbone factory. If you want to add your own backbone: 1. copy a previous backbone file and rename (e.g. backboneExample.py), 2. make sure that your backbone class inherits from 'Backbone' (e.g. ```class backboneExample(Backbone)``` and ```super(backboneExample, self).__init__()``` in the class init. and add the following ```if __name__ == "__main__":
      print(Backbone._backbone_registry)```, 3. import the class in backbone_factory.py and add to the registered_backbone_classes.
    - ```/keypoint_detection/models/detector.py```: This file contains the main code that creates a head and a backbone, performs the training, logging and glue code between other modules.
    - ```/keypoint_detection/models/metrics.py```: contains the code that performs the AP metrics. I added a function ```keypoint_classification_OKS``` which  implements the OKS metric. If you want to use the old metric by Thomas Lips which uses simple euclidean distance to assigen FP and TP, uncomment the old code in the ```update``` function and comment the OKS part.
  - ```/keypoint_detection/train```: contains ```train.py``` which is the file you want to run (with args) to start training a model. Folder also contains ```utils.py```
  - ```/keypoint_detection/utils```:
    - ```/keypoint_detection/utils/heatmap.py```: contains a function that generates a heatmap and a function that gets keypoints from a heatmap.
    - ```/keypoint_detection/utils/load_checkpoints.py```: code that allows starting training from a checkpoint. Used by ```detector.py``` if a path to a checkpoint is given when running ```train.py```.
    - ```/keypoint_detection/utils/visualization.py```: code used to visualize predictions, for logging purposes.

- ```/scripts``` : Scripts for benchmarking, training (e.g. back_to_back), and inference can be found in this folder.

- ```/test```: Contains test files created by Thomas Lips.

- ```/labeling```: Contains tools for labeling and conversion of data. I created ```/labeling/scripts/resize_coco_dataset.py``` to resize the resolution of COCO datasets. (To allow faster training)

## Models
Results from experiments:

**Model (BCE)**

| Model            | mAP    | AP50   | AP75   | Parameters (M) | Inference (ms) | FPS     |
|------------------|--------|--------|--------|----------------|----------------|---------|
| MaxVitUnet       | 0.2981 | 0.4827 | 0.2856 | 16.8           | 41.027         | 24.374  |
| ConvNeXt-nano    | 0.287  | 0.4802 | 0.2708 | 17.7           | 41.306         | 24.210  |
| Hourglass-2-5    | 0.2769 | 0.4496 | 0.257  | 28.9           | 26.739         | 37.399  |
| ConvNeXt-femto   | 0.2763 | 0.4673 | 0.2631 | 5.8            | 20.702         | 48.305  |
| Hourglass-4-2    | 0.2722 | 0.3951 | 0.2423 | 1.4            | 35.997         | 27.780  |
| Hourglass-2-4    | 0.2705 | 0.4412 | 0.2466 | 7.4            | 24.131         | 41.440  |
| Hourglass-1-4    | 0.2595 | 0.4305 | 0.2499 | 3.6            | 14.428         | 69.310  |
| MobileNetV3      | 0.254  | 0.4339 | 0.2384 | 3.4            | 16.629         | 60.136  |
| Unet-4-2         | 0.2295 | 0.4145 | 0.2078 | 0.13           | 14.429         | 69.305  |
| DilatedCnn       | 0.2254 | 0.3969 | 0.2054 | 0.102          | 31.425         | 31.822  |
| S3K              | 0.2201 | 0.3968 | 0.1986 | 0.113          | 13.875         | 72.072  |
| Unet-3-2         | 0.22   | 0.3804 | 0.2019 | 0.112          | 13.911         | 71.886  |

**Model (MSE)**

| Model            | mAP    | AP50   | AP75   | Parameters (M) | Inference (ms) | FPS     |
|------------------|--------|--------|--------|----------------|----------------|---------|
| MaxVitUnet       | 0.3033 | 0.4803 | 0.2912 | 16.8           | 41.027         | 24.374  |
| ConvNeXt-nano    | 0.2807 | 0.4461 | 0.2428 | 17.7           | 41.306         | 24.210  |
| MobileNetV3      | 0.274  | 0.4476 | 0.263  | 3.4            | 16.629         | 60.136  |
| ConvNeXt-femto   | 0.2702 | 0.441  | 0.2536 | 5.8            | 20.702         | 48.305  |
| Hourglass-1-4    | 0.2698 | 0.4448 | 0.2553 | 3.6            | 14.428         | 69.310  |
| Unet-4-2         | 0.2592 | 0.4177 | 0.2485 | 0.13           | 14.429         | 69.305  |
| S3K              | 0.2466 | 0.3969 | 0.2336 | 0.113          | 13.875         | 72.072  |
| Unet-3-2         | 0.2396 | 0.4023 | 0.2252 | 0.112          | 13.911         | 71.886  |

**Model (SmoothL1)**

| Model            | mAP    | AP50   | AP75   | Parameters (M) | Inference (ms) | FPS     |
|------------------|--------|--------|--------|----------------|----------------|---------|
| MaxVitUnet       | 0.2981 | 0.4735 | 0.2891 | 16.8           | 41.027         | 24.374  |
| ConvNeXt-nano    | 0.2753 | 0.4199 | 0.2476 | 17.7           | 41.306         | 24.210  |
| MobileNetV3      | 0.2718 | 0.4483 | 0.2562 | 3.4            | 16.629         | 60.136  |
| Hourglass-1-4    | 0.2678 | 0.4409 | 0.2547 | 3.6            | 14.428         | 69.310  |
| Unet-4-2         | 0.2622 | 0.4337 | 0.2503 | 0.13           | 14.429         | 69.305  |
| Unet-3-2         | 0.2538 | 0.4189 | 0.2395 | 0.112          | 13.911         | 71.886  |
| S3K              | 0.2407 | 0.3937 | 0.2255 | 0.113          | 13.875         | 72.072  |

## How to train
To start training, first follow the installation instructions by Thomas Lips below. Then make sure that you've activated your environment by using ```conda activate keypoint-detection```. After this create a ```.sh``` file, e.g. in the ```/scripts``` folder. Bellow is an example train script that, when created, can be used by running ```keypoint-detection$ bash scripts/train.sh```.

```bash
#!/bin/bash
python keypoint_detection/train/train.py\
--keypoint_channel_configuration "keypoint" \
--wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--json_dataset_img_size "512x512" --batch_size 4 --seed 2023 \
--max_epochs 40 --early_stopping_relative_threshold 0.0001 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "MaxVitUnet" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_1r_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --variable_heatmap_sigma 2 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "BCE"
```
This example contains most of the important args that can be used for train.py. Now follows a comprehensive description and notes for some of these arguments. For a complete overview run ```python train/train.py -h```.
- keypoint_channel_configuration: A list of keypoint that need to be learned, channels seperated by a ```;``` and categories within a channel with a ```=```. Best understood by some examples:
  - face: ```"left_eye; right_eye; nose; mouth"``` (eyes can be differentiated from eachother)
  - plant: ```"stem; leaf0= leaf1 = leaf2"``` (leafs can't be differentiated from eachother, but need to detected, thus use same (heatmap) channel but 3 categories)
  - cardboard box: ```"box_corner0 = box_corner1 = box_corner2 = box_corner3; flap_corner; flap_corner2"``` (box corners can't be differentiated from eachother)
  - something with only 1 keypoint: ```"keypoint"```
- wandb_project & wandb_entity: after logging in with an account that has access to these (run `wandb login`), this will be the project where the results will be logged.
- json_dataset_path: Location of the train dataset, here asumes 1. this repository is installed in mono/python/vtc_keypoint/vtc_keypoint and 2. dataset is installed in mono/projects/Agriplanter/AGP_PPS/data/dataset
- json_dataset_img_size: This string is added after data/dataset path, used to easily change which resolution of data to be used (first these datasets needs to be created!). E.g.: ```/data/dataset/train.json``` becomes ```/data/dataset_512x512/train.json``` - batch_size: determines batch_size, to high batch size will result in errors since GPU won't be able to handle it.
- maximal_gt_keypoint_pixel_distances: values for which the AP score needs to be calculated for, when using OKS ```"0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95"``` is commonly used. For using the old euclidean distance ```"2 4"``` is recommended. - n_channels_in: The number of color channels of the images.
- heatmap_sigma: The sigma used for the ground-truth heatmap blobs. Higher sigma results in faster but inprecise training, lower sigma results in slower (or impossible) but precise training)
- variable_heatmap_sigma: If set to a value lower the heatmap_sigma, the heatmap sigma wil start lowering after the 10th epoch, this allows the model to learn fast and then get more precise allong the way.
- n_resnet_blocks, n_downsampling_layers: Values to determine U-Net depth
- n_hourglasses, n_hg_blocks : Values to determine number of stacked hourglasses and depth of the hourglasses.
- augment_train: If added it will train with image augmentations (rotation, color, flip)
- loss_function: Determines the function used to calculate loss, choose between BCE, MSE, SmoothL1
<h1></h1>









<h1 align="center">Thomas Lips Pytorch Keypoint Detection</h1>

This repo contains a Python package for 2D keypoint detection using [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) and [wandb](https://docs.wandb.ai/). Keypoints are trained using Gaussian Heatmaps, as in [Jakab et Al.](https://proceedings.neurips.cc/paper/2018/hash/1f36c15d6a3d18d52e8d493bc8187cb9-Abstract.html) or [Centernet](https://github.com/xingyizhou/CenterNet).

This package is been used for research at the [AI and Robotics](https://airo.ugent.be/projects/computervision/) research group at Ghent University. You can see some applications below: The first image shows how this package is used to detect corners of cardboard boxes, in order to close the box with a robot. The second example shows how it is used to detect a varying number of flowers.
<div align="center">
  <img src="doc/img/box-keypoints-example.png" width="80%">
  <img src="doc/img/keypoints-flowers-example.png" width="80%">
</div>


## Main Features

- This package contains **different backbones** (Unet-like, dilated CNN, Unet-like with pretrained ConvNeXt encoder). Furthermore you can  easily add new backbones or loss functions. The head of the keypoint detector is a single CNN layer.
- The package uses the often-used **COCO dataset format**.
- The detector can deal with an **arbitrary number of keypoint channels**, that can contain **a varying amount of keypoints**. You can easily configure which keypoint types from the COCO dataset should be mapped onto the different channels of the keypoint detector.
- The package contains an implementation of the Average Precision metric for keypoint detection.
- Extensive **logging to wandb is provided**: The loss for each channel is logged, together with the AP metrics for all specified treshold distances. Furthermore, the raw heatmaps, detected keypoints and ground truth heatmaps are logged at every epoch for the first batch to provide insight in the training dynamics and to verify all data processing is as desired.
- All **hyperparameters are configurable** using a python argumentparser or wandb sweeps.

note: this is the second version of the package, for the older version that used a custom dataset format, see the github releases.


TODO: add integration example.

## Local Installation
- clone this repo in your project (e.g. as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules), using [vcs](https://github.com/dirk-thomas/vcstool),..). It is recommended to lock to the current commit as there are no guarantees w.r.t. backwards comptability.
- create a conda environment using `conda env create --file environment.yaml`
- activate with `conda activate keypoint-detection`
- run `wandb login` to set up your wandb account.
- you are now ready to start training.

## Dataset

This package used the [COCO format](https://cocodataset.org/#format-data) for keypoint annotation and expects a dataset with the following structure:
```
dataset/
  images/
    ...
  <name>.json : a COCO-formatted keypoint annotation file.
```
For an example, see the `test_dataset` at `test/test_dataset`.


### Labeling
If you want to label data, we provide integration with the [CVAT](https://github.com/opencv/cvat) labeling tool: You can annotate your data and export it in their custom format, which can then be converted to COCO format. Take a look [here](labeling/Readme.md) for more information on this workflow and an example. To visualize a given dataset, you can use the  `keypoint_detection/utils/visualization.py` script.

## Training

There are 2 ways to train the keypoint detector:

- The first is to run the `train.py` script with the appropriate arguments. e.g. from the root folder of this repo, you can run the bash script `bash test/integration_test.sh` to test on the provided test dataset, which contains 4 images. You should see the loss going down consistently until the detector has completely overfit the train set and the loss is around the entropy of the ground truth heatmaps (if you selected the default BCE loss).

- The second method is to create a sweep on [wandb](https://wandb.ai) and to then start a wandb agent from the correct relative location.
A minimal sweep example  is given in `test/configuration.py`. The same content should be written to a yaml file according to the wandb format. The sweep can be started by running `wandb agent <sweep-id>` from your CLI.


To create your own configuration: run `python train.py -h` to see all parameter options and their documentation.

## Using a trained model (Inference)
During training Pytorch Lightning will have saved checkpoints. See `scripts/checkpoint_inference.py` for a simple example to run inference with a checkpoint.
For benchmarking the inference (or training), see `scripts/benchmark.py`.

## Development  info
- formatting and linting is done using [pre-commit](https://pre-commit.com/)
- testing is done using pytest (with github actions for CI)


## Note on performance
- Keep in mind that the Average Precision is a very expensive operation, it can easily take as long to calculate the AP of a .1 data split as it takes to train on the remaining 90% of the data. Therefore it makes sense to use the metric sparsely. The AP will always be calculated at the final epoch, so for optimal train performance (w/o intermediate feedback), you can e.g. set the `ap_epoch_start` parameter to your max number of epochs + 1.
