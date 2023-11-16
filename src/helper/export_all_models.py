from ..models.lightning import LightningWrapper
import torch


def main():
    BATCH_SIZE = 1
    image_size = (480, 640)
    num_classes = 5

    variants = [
        "squeezenet",
        # "mobilenetv3",
        # "mobilenetv2",
    ]

    for variant in variants:
        model = LightningWrapper(
            image_size,
            num_classes,
            model=variant,
            batch_size=BATCH_SIZE,
            iou_threshold=0.5,
            conf_threshold=0.2,
            detections_per_img=50,
        )
        image_size = model.image_size
        input_image = torch.randn((1, 3, *image_size))
        name = f"{model.model_variant}_{model.image_size[0]}_{model.image_size[1]}.onnx"
        torch.onnx.export(
            model.model,
            input_image,
            name,
            input_names=["data"],
            output_names=["output"],
        )


if __name__ == "__main__":
    main()
