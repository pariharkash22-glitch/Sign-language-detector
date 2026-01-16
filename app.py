import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import mediapipe as mp
import pickle
from PIL import Image, ImageTk
from datetime import datetime

class SignLanguageGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detector")
        self.root.geometry("700x600")

        # Load trained model
        try:
            self.model = pickle.load(open('model.p', 'rb'))
        except:
            messagebox.showerror("Error", "model.p not found! Run 02_train.py first.")

        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7)
        
        # UI Elements
        tk.Label(root, text="Sign Language System (6PM-10PM)", font=("Arial", 14, "bold")).pack(pady=10)
        
        self.btn_upload = tk.Button(root, text="Upload Image", command=self.upload_image, width=20)
        self.btn_upload.pack(pady=5)

        self.btn_live = tk.Button(root, text="Start Real-time", command=self.start_live, width=20)
        self.btn_live.pack(pady=5)

        self.display_label = tk.Label(root)
        self.display_label.pack(pady=10)

        self.result_label = tk.Label(root, text="Result: None", font=("Arial", 18), fg="blue")
        self.result_label.pack(pady=10)

    def is_valid_time(self):
        # Restriction: 6 PM (18) to 10 PM (22)
        hour = datetime.now().hour
        return 18 <= hour < 22

    def predict_sign(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_rgb)
        if results.multi_hand_landmarks:
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return self.model.predict([landmarks])[0]
        return "No hand detected"

    def upload_image(self):
        if not self.is_valid_time():
            messagebox.showwarning("Locked", "System only works from 6 PM to 10 PM.")
            return

        path = filedialog.askopenfilename()
        if path:
            frame = cv2.imread(path)
            prediction = self.predict_sign(frame)
            self.result_label.config(text=f"Result: {prediction}")
            
            # Show image
            img = cv2.resize(frame, (400, 300))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.display_label.config(image=img)
            self.display_label.image = img

    def start_live(self):
        if not self.is_valid_time():
            messagebox.showwarning("Locked", "System only works from 6 PM to 10 PM.")
            return
        
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            prediction = self.predict_sign(frame)
            self.result_label.config(text=f"Result: {prediction}")
            
            # Display live feed
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.display_label.config(image=img)
            self.display_label.image = img
            self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageGUI(root)
    root.mainloop()