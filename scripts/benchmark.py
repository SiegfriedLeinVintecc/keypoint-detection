import torch


def benchmark(f, name=None, iters=500, warmup=20, display=True, profile=False):
    """Pytorch Benchmark script, copied from Horace He at https://gist.github.com/Chillee/f86675147366a7a0c6e244eaa78660f7#file-2-overhead-py"""
    import time

    # warmup as some operations are initialized lazily
    # or some optimizations still need to happen
    for _ in range(warmup):
        f()
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    torch.cuda.synchronize()  # wait for all kernels to finish
    begin = time.time()
    for _ in range(iters):
        f()
    torch.cuda.synchronize()  # wait for all kernels to finish
    us_per_iter = (time.time() - begin) * 1e6 / iters
    if name is None:
        res = us_per_iter
    else:
        res = f"{name}: {us_per_iter:.2f}us / iter"
    if display:
        print(res)
    return res


if __name__ == "__main__":
    """example code for benchmarking model/inference speed"""
    import numpy as np
    from checkpoint_inference import local_inference

    from keypoint_detection.models.backbones.backbone_factory import BackboneFactory
    from keypoint_detection.models.detector import KeypointDetector

    # registered_backbone_classes: List[Backbone] = [Unet, ConvNeXtUnet, MaxVitUnet, S3K, DilatedCnn, MobileNetV3, Hourglass]
    device = "cuda:0"
    input_width = 512
    input_height = 512

    sample_model_input = torch.rand(1, 3, input_width, input_height, device=device, dtype=torch.float32)
    sample_inference_input = np.random.randint(0, 255, (input_width, input_height, 3), dtype=np.uint8)

    backbones = ["Unet", "ConvNeXtUnet", "MaxVitUnet", "S3K", "DilatedCnn", "MobileNetV3", "Hourglass"]

    for bb in backbones:
        backbone = BackboneFactory.create_backbone(bb)
        model = KeypointDetector(1, "2 4", 3, 4e-4, backbone, [["keypoint"]], 1, 1, 0.0, 20, 2.0)
        model.eval()
        model.to(device)
        print("---------------", bb, "----------------")
        benchmark(
            lambda: local_inference(model, sample_inference_input, device=device), "plain model inference", profile=False
        )
        benchmark(
            lambda: local_inference(model, sample_inference_input, device=device), "plain model inference", profile=False
        )
