import cv2
import numpy as np
import pillow_heif
import pymupdf
from PIL import Image

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pillow:", Image.__version__)
print("pillow-heif:", pillow_heif.__version__)
print("PyMuPDF:", pymupdf.__version__)


img = np.zeros((200, 400, 3), dtype=np.uint8)

cv2.putText(img, "God is one", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 15, 259))

cv2.imwrite("output/shahada.png", img)

print("Image shape:", img.shape)
