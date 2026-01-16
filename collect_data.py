import os
import sys

# 1. Suppress TensorFlow logs to stop the 'oneDNN' hang
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2

# 2. Robust MediaPipe Import
try:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    print("--- MediaPipe loaded successfully! ---")
except Exception as e:
    print(f"Installation Error: {e}")
    sys.exit()

# Initialize
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

print("Starting Camera... (Please wait up to 30 seconds for first run)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Process
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    cv2.imshow('Hand Tracking Test', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()