import cv2
import numpy as np

img = cv2.imread(input(), cv2.IMREAD_COLOR)  # img.png
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# uint8 and C-contiguous
gray = np.ascontiguousarray(gray, dtype=np.uint8)

# RAW
with open("img.raw", "wb") as f:
    f.write(gray.tobytes())

# PNG
cv2.imwrite("img.png", gray)
