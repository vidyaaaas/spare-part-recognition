# import cv2
# import numpy as np

# image_path = r"C:\Users\vidya singh\OneDrive\Documents\new\dataset\Part_01\1-images-0.jpg"

# img = cv2.imread(image_path)

# if img is None:
#     print("Error: Could not load image.")
#     print("Check the image path.")
#     exit()

# height, width = img.shape[:2]

# if width > 1200:
#     scale = 1200 / width
#     img = cv2.resize(
#         img,
#         (int(width * scale), int(height * scale))
#     )

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# blur = cv2.GaussianBlur(gray, (5, 5), 0)


# edges = cv2.Canny(blur, 50, 150)


# kernel = np.ones((5, 5), np.uint8)

# edges = cv2.dilate(edges, kernel, iterations=2)
# edges = cv2.erode(edges, kernel, iterations=1)


# contours, _ = cv2.findContours(
#     edges,
#     cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE
# )

# if not contours:
#     print("No spare part detected.")
#     exit()

# largest_contour = max(contours, key=cv2.contourArea)

# if cv2.contourArea(largest_contour) < 500:
#     print("Detected contour is too small.")
#     exit()

# x, y, w, h = cv2.boundingRect(largest_contour)


# padding = 25

# x = max(0, x - padding)
# y = max(0, y - padding)

# w = min(img.shape[1] - x, w + (2 * padding))
# h = min(img.shape[0] - y, h + (2 * padding))

# cropped = img[y:y+h, x:x+w]

# output_path = "cropped_spare_part.jpg"

# cv2.imwrite(output_path, cropped)

# display_img = img.copy()

# cv2.rectangle(
#     display_img,
#     (x, y),
#     (x + w, y + h),
#     (0, 255, 0),
#     2
# )

# cv2.imshow("Original Image", img)
# cv2.imshow("Detected Spare Part", display_img)
# cv2.imshow("Cropped Spare Part", cropped)

# print("Cropping completed successfully.")
# print("Saved as:", output_path)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import numpy as np

image_path = r"C:\Users\vidya singh\OneDrive\Documents\new\dataset\Part_01\1-images-0.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Error: Could not load image.")
    print("Check the image path.")
    exit()

height, width = img.shape[:2]

if width > 1200:
    scale = 1200 / width
    img = cv2.resize(
        img,
        (int(width * scale), int(height * scale))
    )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 50, 150)

kernel = np.ones((5, 5), np.uint8)

edges = cv2.dilate(edges, kernel, iterations=2)
edges = cv2.erode(edges, kernel, iterations=1)

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    print("No spare part detected.")
    exit()

largest_contour = max(contours, key=cv2.contourArea)

if cv2.contourArea(largest_contour) < 500:
    print("Detected contour is too small.")
    exit()

x, y, w, h = cv2.boundingRect(largest_contour)

padding = 25

x = max(0, x - padding)
y = max(0, y - padding)

w = min(img.shape[1] - x, w + (2 * padding))
h = min(img.shape[0] - y, h + (2 * padding))

cropped = img[y:y+h, x:x+w]

output_path = "cropped_spare_part.jpg"

cv2.imwrite(output_path, cropped)

display_img = img.copy()

cv2.rectangle(
    display_img,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    2
)

display_original = cv2.resize(img, (600, 400))
display_detected = cv2.resize(display_img, (600, 400))
display_cropped = cv2.resize(cropped, (600, 400))

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Detected Spare Part", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cropped Spare Part", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Original Image", 600, 400)
cv2.resizeWindow("Detected Spare Part", 600, 400)
cv2.resizeWindow("Cropped Spare Part", 600, 400)

cv2.moveWindow("Original Image", 0, 0)
cv2.moveWindow("Detected Spare Part", 620, 0)
cv2.moveWindow("Cropped Spare Part", 310, 450)

cv2.imshow("Original Image", display_original)
cv2.imshow("Detected Spare Part", display_detected)
cv2.imshow("Cropped Spare Part", display_cropped)

print("Cropping completed successfully.")
print("Saved as:", output_path)

cv2.waitKey(0)
cv2.destroyAllWindows()