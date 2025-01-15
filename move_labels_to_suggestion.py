from pathlib import Path
import json
import sys

datasets = sys.argv[1:]
for dataset in datasets:
    data_json = {}

    dataset_path = Path.cwd().joinpath(dataset)
    json_files = list(dataset_path.joinpath("images").glob("*.json"))

    for json_file in json_files:
        image_file = json_file.with_suffix(".png")
        assert image_file.exists(), f"Image file {image_file} does not exist"
        data_json[image_file.name] = json.load(json_file.open())

    with dataset_path.joinpath("data.json").open("w") as f:
        json.dump(data_json, f)
