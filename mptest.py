import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # print(results)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_world_landmarks:
      for hand_world_landmarks in results.multi_hand_world_landmarks:
        # print('hand_world_landmarks:', hand_world_landmarks)
        # print(
        #   f'Index finger tip coordinates: (',
        #   f'{hand_world_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x}, '
        #   f'{hand_world_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y})'
        # )
        for idx, landmark in enumerate(hand_world_landmarks.landmark):
                print(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}")
        mp_drawing.draw_landmarks(
            image,
            hand_world_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        if cv2.waitKey(1) & 0xFF == ord('f'):
          print('hand_world_landmarks:', hand_world_landmarks)
          print("\n\n\n\n")
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()