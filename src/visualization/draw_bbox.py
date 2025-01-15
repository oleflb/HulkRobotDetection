from matplotlib.patches import Rectangle
from torchvision.ops import nms

CLASSES = ["Background", "Ball", "Robot", "Goal Post", "Penalty Spot", "LSpot", "TSpot", "XSpot"]


def draw_bboxes_on_axis_from_prediction(
    axis, prediction, height, width, confidence_threshold=0.2, color="red"
):  
    box_indices = nms(prediction["boxes"], prediction["scores"], confidence_threshold)

    for bbox, label, score in zip(
        prediction["boxes"][box_indices],
        prediction["labels"][box_indices],
        prediction["scores"][box_indices],
    ):
        if score < confidence_threshold:
            continue
        xl = int(bbox[0])
        yu = int(bbox[1])
        xr = int(bbox[2])
        yd = int(bbox[3])
        axis.add_patch(
            Rectangle(
                (xl, yd), xr - xl, yu - yd, fill=False, linestyle=":", color=color
            )
        )
        axis.text(xl, yu, f"{CLASSES[label]} - {score:.2f}")


def draw_bboxes_on_axis_from_truth(axis, labels, color="green"):
    for bbox, label in zip(labels["boxes"], labels["labels"]):
        xl = int(bbox[0])
        yu = int(bbox[1])
        xr = int(bbox[2])
        yd = int(bbox[3])
        axis.add_patch(Rectangle((xl, yd), xr - xl, yu - yd, fill=False, color=color))
        axis.text(xl, yu, f"{CLASSES[label]}", color=color)
