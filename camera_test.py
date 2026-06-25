import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

window_name = "Live Camera Feed"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        print("Closing camera...")
        break

    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE) < 0:
            print("Window closed.")
            break
    except cv2.error:
        print("Window closed.")
        break

cap.release()
cv2.destroyAllWindows()