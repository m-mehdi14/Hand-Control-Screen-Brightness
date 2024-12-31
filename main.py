import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np

def main():
    # Initialize webcam and Mediapipe Hands
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break

        # Convert the image to RGB format
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        lmList = []
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(handLandmarks.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

        if lmList:
            # Coordinates of thumb tip and index finger tip
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # Draw circles on the thumb and index finger tips
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

            # Draw a line between the two points
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Calculate the distance between the two points
            length = hypot(x2 - x1, y2 - y1)

            # Interpolate brightness based on the distance
            bright = np.interp(length, [15, 220], [0, 100])
            print(f"Brightness: {bright}, Distance: {length}")

            # Set screen brightness (requires permissions on macOS)
            sbc.set_brightness(int(bright))

        # Display the webcam feed
        cv2.imshow("Image", img)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()