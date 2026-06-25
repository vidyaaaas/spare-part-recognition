

import cv2


image_paths = [
    r"C:\Users\vidya singh\OneDrive\Documents\new\dataset\Part_01\1-images-0.jpg",
    r"C:\Users\vidya singh\OneDrive\Documents\new\dataset\Part_01\1-images-1.jpg",
    r"C:\Users\vidya singh\OneDrive\Documents\new\dataset\Part_01\1-images-2.jpg"
]

orb = cv2.ORB_create(
    nfeatures=1500,
    fastThreshold=110
)

for i, image_path in enumerate(image_paths):

    print(f"\nProcessing Image {i+1}")

    
    img = cv2.imread(image_path)

    if img is None:
        print("Image not found:", image_path)
        continue

    
    h, w = img.shape[:2]

    max_width = 1200

    if w > max_width:
        scale = max_width / w
        new_width = int(w * scale)
        new_height = int(h * scale)

        img = cv2.resize(img, (new_width, new_height))

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    
    keypoints = sorted(
        keypoints,
        key=lambda x: x.response,
        reverse=True
    )[:120]

    print("Number of keypoints detected:", len(keypoints))

    
    output = img.copy()

    for kp in keypoints:
        x = int(kp.pt[0])
        y = int(kp.pt[1])

        cv2.circle(
            output,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    
    output_file = f"orb_output_{i+1}.jpg"
    cv2.imwrite(output_file, output)

    print("Saved:", output_file)

    
    display_img = cv2.resize(output, (900, 600))

    cv2.namedWindow("ORB Features", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ORB Features", 1000, 700)
    cv2.imshow("ORB Features", display_img)

    print("Press any key for next image...")
    cv2.waitKey(0)

cv2.destroyAllWindows()