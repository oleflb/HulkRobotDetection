import glob
import sys
import os
from os import path
import json
import numpy as np
from tqdm import tqdm

labels = ["Ball", "Robot", "GoalPost", "PenaltySpot", "LSpot", "TSpot", "XSpot"]
# export = ["Ball", "Robot", "GoalPost", "PenaltySpot"]
export = ["Robot"]

json_paths = glob.glob(path.join(sys.argv[1], "*.json"))


def convert_xyxy_to_cxcywh(xyxy: np.ndarray):
    [minx, miny, maxx, maxy] = xyxy

    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    w = maxx - minx
    h = maxy - miny

    return cx / 640, cy / 480, w / 640, h / 480

for json_path in tqdm(json_paths):
    json_object = json.load(open(json_path))
    txt_lines = []
    for bbox in json_object:
        points = np.array(bbox["points"]).flatten()
        if bbox["class"] not in labels:
            raise ValueError(f"{bbox['class']} is invalid, available classes are {labels}")
        if bbox["class"] not in export:
            continue
        label = export.index(bbox["class"])
        
        cx, cy, w, h = convert_xyxy_to_cxcywh(points)

        txt_lines.append(f"{label} {cx} {cy} {w} {h}\n")

    txt_file = path.splitext(json_path)[0] + ".txt"
    with open(txt_file, "w") as f:
        f.writelines(txt_lines)
