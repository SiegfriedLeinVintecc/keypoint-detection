#!/bin/bash

# make sure to remove all trailing spaces from the command, as this would result in an error when using bash.
python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "S3K" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "ConvNeXtUnet" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "DilatedCnn" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "MaxVitUnet" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "MobileNetV3" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Unet" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 50 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Unet" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 4 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Hourglass" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Hourglass" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 2 --n_hg_blocks 4 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Hourglass" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 4 --n_hg_blocks 2 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 50 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Hourglass" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 2 --n_hg_blocks 5 --augment_train --loss_function "BCE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "S3K" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "ConvNeXtUnet" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "DilatedCnn" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "MaxVitUnet" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "MobileNetV3" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Unet" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 50 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Unet" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 4 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Hourglass" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 1 --n_hg_blocks 4 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Hourglass" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 2 --n_hg_blocks 4 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 40 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Hourglass" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 4 --n_hg_blocks 2 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2; python keypoint_detection/train/train.py \
--keypoint_channel_configuration "keypoint" \
--json_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/train.json" --json_validation_dataset_path "../../../../projects/Agriplanter/AGP_PPS/data/dataset/val.json" \
--batch_size 4 --json_dataset_img_size "512x512" \
--seed 2023 --wandb_project "keypoint-detector-agriplanter-OKS" --wandb_entity "vintecc-siegfried-lein" \
--max_epochs 50 --early_stopping_relative_threshold -1 --log_every_n_steps 1 --accelerator="gpu" --devices 1 --precision 16 \
--backbone_type "Hourglass" --learning_rate 0.0004 --maximal_gt_keypoint_pixel_distances "0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.90 0.95" --ap_epoch_freq 2 \
--auto_lr_find True --fast_dev_run False --n_channels_in 3 --heatmap_sigma 2 --minimal_keypoint_extraction_pixel_distance 10 \
--n_resnet_blocks 3 --n_downsampling_layers 2 --n_hourglasses 2 --n_hg_blocks 5 --augment_train --loss_function "MSE" --variable_heatmap_sigma 1.2;