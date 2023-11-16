import glob
import cv2
import sys


def main():
    print(f"scanning path {sys.argv[1]}")
    files = glob.glob(sys.argv[1])

    print(f"foung {len(files)} images to convert, continue?")
    if input("[Y/n] ") == "n":
        return

    for image_path in files:
        print(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
        cv2.imwrite(image_path, img)


if __name__ == "__main__":
    main()
