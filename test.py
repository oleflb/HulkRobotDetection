from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

model = YOLO("yolov8n-pose.pt")
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"

frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
      print("Video frame is empty or video processing has been successfully completed.")
      break
    frame_count += 1
    results = model.predict(im0, verbose=True)
    keypoints = results[0].keypoints.data
    annotator = Annotator(im0, line_width=2)
    
    for ind, k in enumerate(reversed(keypoints)):
        annotator.kpts(k, shape=(640, 640), radius=1, kpt_line=True)

    cv2.imshow("detections", annotator.result())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     

cv2.destroyAllWindows()