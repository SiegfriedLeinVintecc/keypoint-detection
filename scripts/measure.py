from checkpoint_inference import local_inference
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
import pathlib
from skimage import io
import wandb

if __name__ == "__main__":

    """
    Measure keypoint distance to ground truth to compare
    """
    #checkpoint_name: str e.g. 'airo-box-manipulation/iros2022_0/model-17tyvqfk:v3'

    #checkpoint_name = ["vintecc-siegfried-lein/keypoint-detector-agriplanter/model-ajx7pvmv:v14", "vintecc-siegfried-lein/keypoint-detector-agriplanter/model-snk2hjnu:v5"]
    checkpoint_name = ["vintecc-siegfried-lein/keypoint-detector-agriplanter/model-ri5pmfde:v9", "vintecc-siegfried-lein/keypoint-detector-agriplanter/model-7juvdkdp:v10", "vintecc-siegfried-lein/keypoint-detector-agriplanter/model-ntvkq45u:v22", "vintecc-siegfried-lein/keypoint-detector-agriplanter/model-m7w4pcfj:v7", "vintecc-siegfried-lein/keypoint-detector-agriplanter/model-4nt3b1d0:v36"]
    folder_path = "../../../../projects/Agriplanter/AGP_PPS/data/dataset_512x512/MEURILLION/22_03_24__19_21_00_746932"
    
    # load a wandb checkpoint
    for name in checkpoint_name:

        # download checkpoint locally (if not already cached)
        run = wandb.init(project="inference")
        artifact = run.use_artifact(name, type="model")
        artifact_dir = artifact.download()
    # model.eval()
    # model.cuda()

    # # load image from disk
    # # although at inference you will most likely get the image from the camera
    # # format will be the same though:
    # #  (height, width, channels) ints in range [0,255]
    # # beware of the color channels order, it should be RGBD.
    # for image_path in pathlib.Path(folder_path).iterdir():
    #     if image_path.is_file() and image_path.suffix == ".png":
    #         image_path = pathlib.Path(image_path)
    #         image = io.imread(image_path)

    #         # inference
    #         keypoints = local_inference(model, image)
    #         print(keypoints)