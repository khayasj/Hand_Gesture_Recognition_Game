import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from pynput.keyboard import Key, Controller
import time
import pickle

class GameGestureController:
    def __init__(self, model_path='Hand_Gesture_Recognition.h5', label_encoder_path='label_encoder.pkl'):
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,  
            min_tracking_confidence=0.7    
        )
        
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load the label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Initialize keyboard controller
        self.keyboard = Controller()
        
        # Define gesture to key mappings
        self.gesture_controls = {
            'four': 'x',
            'like': 'z',
            'palm': 'a',
            'mute': 'w',
            'peace': 's',
            'rock' : 'd',
            'stop': Key.esc,
            'ok': Key.enter,
            'two_up': Key.left,
            'one': Key.up,
            'dislike': Key.down,
            'fist': Key.space,
            'three': Key.right
        }
        
        # Gesture smoothing with shorter window
        self.prev_gesture = None
        self.gesture_counter = 0
        self.min_gesture_frames = 1  
        
        # Active keys tracking
        self.active_keys = set()
        
        # Performance optimization
        self.last_process_time = time.time()
        self.process_interval = 1/30  # Process at 30fps
        
        print("Available gestures:", self.label_encoder.classes_)

    def preprocess_landmarks(self, hand_landmarks):
        landmarks_flat = []
        for landmark in hand_landmarks.landmark:
            landmarks_flat.extend([landmark.x, landmark.y, landmark.z])
        return np.array([landmarks_flat])

    def release_all_keys(self):
        for key in self.active_keys:
            self.keyboard.release(key)
        self.active_keys.clear()

    def execute_gesture_action(self, gesture, confidence):
        if gesture in self.gesture_controls and confidence > 0.5:  # Lowered confidence threshold
            key = self.gesture_controls[gesture]
            
            # Only release keys if gesture changed
            if gesture != self.prev_gesture:
                self.release_all_keys()
                self.keyboard.press(key)
                self.active_keys.add(key)
                self.prev_gesture = gesture
        else:
            self.release_all_keys()
            self.prev_gesture = None

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            while cap.isOpened():
                # Time-based frame processing
                current_time = time.time()
                if current_time - self.last_process_time < self.process_interval:
                    continue
                
                success, image = cap.read()
                if not success:
                    print("Failed to capture frame")
                    continue

                # Process frame
                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Quick prediction
                    landmarks_processed = self.preprocess_landmarks(hand_landmarks)
                    prediction = self.model.predict(landmarks_processed, verbose=0)
                    predicted_idx = np.argmax(prediction[0])
                    confidence = prediction[0][predicted_idx]
                    
                    predicted_gesture = self.label_encoder.inverse_transform([predicted_idx])[0]
                    self.execute_gesture_action(predicted_gesture, confidence)
                    
                    # Display info
                    cv2.putText(image, f"Gesture: {predicted_gesture}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
                    cv2.putText(image, f"Confidence: {confidence:.2f}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
                else:
                    self.release_all_keys()

                cv2.imshow('Game Gesture Controller', image)
                self.last_process_time = current_time
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.release_all_keys()
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()

if __name__ == "__main__":
    controller = GameGestureController()
    controller.run()