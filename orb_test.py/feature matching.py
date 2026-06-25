
import cv2

img1 = cv2.imread(r"C:\Users\vidya singh\OneDrive\Documents\new\dataset\Part_01\1-images-0.jpg")
img2 = cv2.imread(r"C:\Users\vidya singh\OneDrive\Documents\new\dataset\Part_01\1-images-1.jpg")

if img1 is None or img2 is None:
    print("One or both images not found!")
    exit()

img1 = cv2.resize(img1, (800, 600))
img2 = cv2.resize(img2, (800, 600))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=1000)


kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.knnMatch(des1, des2, k=2)


good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)


good_matches = sorted(good_matches, key=lambda x: x.distance)


good_matches = good_matches[:50]

print("Good Matches:", len(good_matches))


matched_img = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite("feature_matching_result.jpg", matched_img)


display = cv2.resize(matched_img, (1400, 700))

cv2.namedWindow("ORB Feature Matching", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ORB Feature Matching", 1400, 700)
cv2.imshow("ORB Feature Matching", display)

cv2.waitKey(0)
cv2.destroyAllWindows()