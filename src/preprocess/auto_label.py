from ultralytics import YOLO
from pathlib import Path
import sys
import json
from tqdm import tqdm

convert_class = {
    "robot": "Robot",
    "ball": "Ball",
    "goal_post": "GoalPost",
    "pen_spot": "PenaltySpot",
}

def main():
    yolo_model = YOLO(sys.argv[1])
    image_folder = Path(sys.argv[2])

    images = list(image_folder.glob("*.png"))
    print(f"Label {len(images)} images")

    for image in tqdm(images):
        labels = []
        results = yolo_model(image)[0]
        class_names = results.names
        for box in results.boxes:
            class_name = class_names[box.cls]
            position = box.xyxy.reshape(2,2).tolist()
            this = {}
            this["points"] = position
            this["class"] = convert_class[class_name]
            labels.append(this)
        label_file = image.with_suffix(".json")
        label_file.write_text(json.dumps(labels))

if __name__ == "__main__":
    main()