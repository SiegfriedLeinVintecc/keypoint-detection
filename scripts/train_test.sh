#!/bin/bash

# make sure to remove all trailing spaces from the command, as this would result in an error when using bash.
python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 60 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Hourglass" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "2 4" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 4 --n_downsampling_layers 3 --n_hourglasses 2 --n_hg_blocks 3 --augment_train --loss_function "BCE" --variable_heatmap_sigma False;