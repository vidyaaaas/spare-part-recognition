
import cv2

image_paths = [
    r"C:\Users\vidya singh\OneDrive\Documents\new\dataset\Part_01\1-images-0.jpg",
    r"C:\Users\vidya singh\OneDrive\Documents\new\dataset\Part_01\1-images-1.jpg",
    r"C:\Users\vidya singh\OneDrive\Documents\new\dataset\Part_01\1-images-2.jpg"
]


orb = cv2.ORB_create()

for image_path in image_paths:

    print("\nProcessing:", image_path)

    img = cv2.imread(image_path)

    if img is None:
        print("Image not found!")
        continue

    print("Image Shape:", img.shape)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    print("Number of keypoints detected:", len(keypoints))

    output = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


    original_resized = cv2.resize(img, (500, 500))
    output_resized = cv2.resize(output, (500, 500))

    cv2.imshow("Original Image", original_resized)
    cv2.imshow("ORB Keypoints", output_resized)

    print("Press any key to see the next image...")
    cv2.waitKey(0)

cv2.destroyAllWindows()