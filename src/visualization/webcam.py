from ..models.ssdnet import SSDNet, ReparameterizedSSDNet
from ..models.lightning import LightningWrapper
import sys
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2 
import numpy as np
from ultralytics.utils.plotting import Annotator
from torchvision.ops import nms

CLASSES = ["Background", "Ball", "Robot", "Goal Post", "Penalty Spot"]

model = LightningWrapper.load_from_checkpoint(
    sys.argv[1], map_location=torch.device("cpu")
)

(h,w) = model.image_size
transform = A.Compose([
    A.Resize(h, w, interpolation=1),
    ToTensorV2(),
])
model = model.model.reparameterize((h,w))
model.train(False)

sx = 640 / w
sy = 480 / h
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  

cv2.namedWindow('detections', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("detections", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    image = frame.astype(np.float32) / 255.0
    image = transform(image=image)["image"]

    detection = model(torch.tensor(image[None, :, :, :]))
    boxes = np.array([prediction["boxes"].detach().numpy() for prediction in detection])[0]
    scores = np.array([prediction["scores"].detach().numpy() for prediction in detection])[0]
    results = ReparameterizedSSDNet.parse_output(torch.tensor(boxes[None, :, :]), torch.tensor(scores), conf_thresh=0.001)[0]

    indices = nms(results["boxes"], results["scores"], iou_threshold=0.2)

    for index in indices:
        box = results["boxes"][index]
        label = results["labels"][index]
        score = results["scores"][index]
        if score < 0.1:
            continue
        x0 = sx*box[0]
        x1 = sx*box[2]
        y0 = sy*box[1]
        y1 = sy*box[3]
        
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