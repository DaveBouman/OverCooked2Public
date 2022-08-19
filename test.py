import numpy as np
import cv2
from PIL import ImageGrab

while True:
    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))  # x, y, w, h
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0Xff == ord('q'):
        break

cv2.destroyAllWindows()
