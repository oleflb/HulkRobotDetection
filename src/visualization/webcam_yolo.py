import sys
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2 
import numpy as np
import torch
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO

CLASSES = ["Background", "Ball", "Robot", "Goal Post", "Penalty Spot"]

model = YOLO(sys.argv[1])

transform = A.Compose([
    A.Resize(96, 128, interpolation=1),
    ToTensorV2(),
])
sx, sy = (640 / 128, 480 / 96)
# define a video capture object 
vid = cv2.VideoCapture(0) 

cv2.namedWindow('detections', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("detections", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    image = frame.astype(np.float32) / 255.0
    boxes = model(transform(image=image)["image"][None, :, :, :])[0].boxes

    for box in boxes:
        score = box.conf.item()
        label = int(box.cls) + 1

        if score < 0.1:
            continue
        xyxy = torch.squeeze(box.xyxy)
        x0 = xyxy[0] * sx
        y0 = xyxy[1] * sy
        x1 = xyxy[2] * sx
        y1 = xyxy[3] * sy
        
        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))
        cv2.rectangle(frame, start_point, end_point, color=(0,255,0), thickness=2)
        cv2.putText(frame, f"{CLASSES[label]} - {score:.2f}", start_point, cv2.FONT_HERSHEY_SIMPLEX , fontScale=1, color=(0,0,0))


    # Display the resulting frame 
    cv2.imshow('detections', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 