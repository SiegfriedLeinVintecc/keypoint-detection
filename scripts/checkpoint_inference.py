"""example script for inference on local image with a saved model checkpoint"""

import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
import time

def local_inference(model, image: np.ndarray, device="cuda"):
    """inference on a single image as if you would load the image from disk or get it from a camera.
    Returns a list of the extracted keypoints for each channel.


    """
    # assert model is in eval mode! (important for batch norm layers)
    assert model.training == False, "model should be in eval mode for inference"

    # convert image to tensor with correct shape (channels, height, width) and convert to floats in range [0,1]
    # add batch dimension
    # and move to device
    image = to_tensor(image).unsqueeze(0).to(device)
    # pass through model
    with torch.no_grad():
        heatmaps = model(image).squeeze(0)

    # extract keypoints from heatmaps
    predicted_keypoints = [
        torch.tensor(get_keypoints_from_heatmap(heatmaps[i].cpu(), 2)) for i in range(heatmaps.shape[0])
    ]

    return predicted_keypoints


if __name__ == "__main__":
    import pathlib

    from skimage import io

    """example for loading models to run inference from a pytorch lightning checkpoint

    for faster inference you probably want to consider
    - reducing the model precision (mixed precision or simply half precision) on new GPUs with TensorCores
            but do not forget to set the cudnn benchmark https://github.com/pytorch/pytorch/issues/46377
    - compiling the model to torchscript
    - or compiling it with Pytorch 2.0 (not yet released)
    - or using TensorRT

    see benchmark.py script for how to test the influences inference speed, which is ofc determined by the model (size), the input size and your hardware.
    """
    #checkpoint_name: str e.g. 'airo-box-manipulation/iros2022_0/model-17tyvqfk:v3'

    checkpoint_name = "vintecc-siegfried-lein/keypoint-detector-agriplanter/model-t4rajt6g:v16"
    folder_path = "../../../../projects/Agriplanter/AGP_PPS/data/dataset_512x512/MEURILLION/22_03_24__19_21_00_746932"

    # load a wandb checkpoint
    model = get_model_from_wandb_checkpoint(checkpoint_name)
    
    # do not forget to set model to eval mode!
    # this will e.g. use the running statistics for batch norm layers instead of the batch statistics.
    # this is important as inference batches are typically a lot smaller which would create too much noise.
    model.eval()

    # move model to gpu
    model.cuda()

    # load image from disk
    # although at inference you will most likely get the image from the camera
    # format will be the same though:
    #  (height, width, channels) ints in range [0,255]
    # beware of the color channels order, it should be RGBD.
    inference_time = 0
    for image_path in pathlib.Path(folder_path).iterdir():
        if image_path.is_file() and image_path.suffix == ".png":
            image_path = pathlib.Path(image_path)
            image = io.imread(image_path)

            start_time = time.time()
            keypoints = local_inference(model, image)
            end_time = time.time()
            print("inference time:", end_time - start_time)
            inference_time = inference_time + (end_time - start_time)
            print(keypoints)
    print("time: ", inference_time)