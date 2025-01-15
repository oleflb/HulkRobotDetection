import onnxruntime as ort
import sys
from ..dataloader.lightningdataset import DataModule
from torchvision.ops import nms
import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

map_metric = MeanAveragePrecision()

model = ort.InferenceSession(sys.argv[1])

print("Inputs:")
for input in model.get_inputs():
    print(f'"{input.name}": {input.shape}')

print("\nOutputs:")
for output in model.get_outputs():
    print(f'"{output.name}": {output.shape}')

main_input = model.get_inputs()[0]
main_output = model.get_outputs()[0]

image_size = main_input.shape[2:]

datamodule = DataModule(image_size, batch_size=1, num_workers=8)
datamodule.setup("validate")

dataloader = datamodule.val_dataloader()

for images, labels in dataloader:
    predictions = model.run(None, {main_input.name: images.numpy()})[0]
    batched_bboxes_cxcywh = torch.from_numpy(predictions[:, :4])
    batched_scores = torch.from_numpy(predictions[:, 4:])
    
    detections = []
    
    for bboxes, scores in zip(batched_bboxes_cxcywh, batched_scores):
        max_score, class_index = torch.max(scores, dim=0)

        scores = max_score[max_score > 0.5]
        indices = class_index[max_score > 0.5]
        bboxes = bboxes[:, max_score > 0.5]


        print(bboxes.shape)
        bboxes_xyxy = torch.cat(
            [
                bboxes[:2, :] - bboxes[2:, :] / 2,
                bboxes[:2, :] + bboxes[2:, :] / 2,
            ],
            dim=1,
        )
        if scores.numel() == 0:
            detections.append({
                "scores": torch.tensor([]),
                "labels": torch.tensor([]),
                "boxes": torch.reshape(torch.tensor([]), (-1, 4)),
            })
            continue

        box_indices = nms(bboxes_xyxy, scores, 0.5)

        scores = scores[box_indices]
        indices = indices[box_indices]
        bboxes = bboxes_xyxy[:, box_indices]

        detections.append({
            "scores": scores.T,
            "labels": indices.T,
            "boxes": bboxes.T,
        })

    map_metric.update(detections, labels)

print(map_metric.compute())


