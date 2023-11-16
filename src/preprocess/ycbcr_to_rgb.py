import glob
import cv2
import sys
from uuid import uuid4
from tqdm import tqdm

glob_paths = sys.argv[1:]
files = list(file for path in glob_paths for file in glob.glob(path))

if input(f"Convert {len(files)} images from ycbcr to rgb into local folder images? [Y/N] ") != "Y":
    print("Aborting...")
    quit()

for image_path in tqdm(files):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    cv2.imwrite(f"./images/{uuid4()}.png", img)