import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

window_name = "Spare Part Camera"
cv2.namedWindow(window_name)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite("captured_part.jpg", frame)
        print("Image saved as captured_part.jpg")

    elif key == ord('q'):
        break

    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 0:
            break
    except:
        break

cap.release()
cv2.destroyAllWindows()