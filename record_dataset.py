import cv2
import os

gesture = input("Enter gesture name: ")

path = f"dataset/{gesture}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

print("Press S to save image")
print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Recording Gesture", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        img_name = f"{path}/{count}.jpg"
        cv2.imwrite(img_name, frame)
        print("Saved:", img_name)
        count += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()