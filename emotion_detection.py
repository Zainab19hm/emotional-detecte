# ============================================================
# Real-Time & Image Emotion Detection System
# Using: OpenCV + Haar Cascades + DeepFace
# Interface: Interactive Tkinter GUI (Camera + Upload Image)
# ============================================================

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from deepface import DeepFace

# ── Load Haar Cascade ─────────────────────────────────────────
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'haarcascade_frontalface_default.xml')
    if os.path.exists(local_path):
        cascade_path = local_path

face_cascade = cv2.CascadeClassifier(cascade_path)

# ── Emotion Dictionaries & Colors ─────────────────────────────
# Colors in BGR for OpenCV
emotion_bgr = {
    'happy':    (46, 204, 113),   # Green
    'sad':      (219, 152, 52),   # Blue
    'angry':    (60, 76, 231),    # Red
    'surprise': (15, 196, 241),   # Yellow
    'fear':     (182, 89, 155),   # Purple
    'disgust':  (34, 126, 230),   # Orange
    'neutral':  (166, 165, 149)   # Gray
}

# Hex colors for Tkinter UI
emotion_hex = {
    'happy':    '#2ecc71',
    'sad':      '#3498db',
    'angry':    '#e74c3c',
    'surprise': '#f1c40f',
    'fear':     '#9b59b6',
    'disgust':  '#e67e22',
    'neutral':  '#95a5a6'
}

emotion_arabic = {
    'happy':    'سعيد 😊',
    'sad':      'حزين 😢',
    'angry':    'غاضب 😠',
    'surprise': 'مندهش 😲',
    'fear':     'خائف 😨',
    'disgust':  'اشمئزاز 🤢',
    'neutral':  'محايد 😐'
}

# ============================================================
# Helper Functions for Live Camera Feed
# ============================================================
def draw_emotion_bar(frame, emotions_dict, x, y, w):
    """Draw emotion percentage bars beside the face bounding box"""
    bar_x = x + w + 10
    bar_y = y
    bar_width = 130
    bar_height = 14
    gap = 4

    for emotion, score in emotions_dict.items():
        color = emotion_bgr.get(emotion, (255, 255, 255))
        filled = int((score / 100) * bar_width)

        # Bar background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
        # Filled part
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_height), color, -1)
        # Label & percentage
        label = f"{emotion}: {score:.1f}%"
        cv2.putText(frame, label, (bar_x + bar_width + 6, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        bar_y += bar_height + gap

def draw_info_panel(frame, emotion, confidence, face_count):
    """Draw top header info overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "Real-Time Emotion Detection", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 255), 2)

    color = emotion_bgr.get(emotion, (255, 255, 255))
    ar_name = emotion_arabic.get(emotion, '')
    cv2.putText(frame, f"Emotion: {emotion.upper()} ({confidence:.1f}%)", (12, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(frame, f"Faces: {face_count}", (w - 110, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, "Press Q to exit", (w - 130, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

def run_live_camera(on_close_callback=None):
    """Run real-time webcam feed with emotion detection"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open webcam. Make sure no other app is using it.")
        if on_close_callback:
            on_close_callback()
        return

    current_emotion = "neutral"
    emotion_confidence = 0.0
    all_emotions = {}
    frame_counter = 0
    ANALYZE_EVERY = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            # Analyze emotion every N frames
            if frame_counter % ANALYZE_EVERY == 0 and len(faces) > 0:
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    current_emotion = result[0]['dominant_emotion']
                    all_emotions = result[0]['emotion']
                    emotion_confidence = float(all_emotions.get(current_emotion, 0))
                except Exception:
                    pass

            # Draw on detected faces
            for (x, y, w, h) in faces:
                color = emotion_bgr.get(current_emotion, (0, 255, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                label = f"{current_emotion.upper()}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), color, -1)
                cv2.putText(frame, label, (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if all_emotions:
                    draw_emotion_bar(frame, all_emotions, x, y, w)

            draw_info_panel(frame, current_emotion, emotion_confidence, len(faces))
            cv2.imshow('Real-Time Emotion Detection - [Press Q to Return]', frame)

            # Check if Q pressed or OpenCV window closed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty('Real-Time Emotion Detection - [Press Q to Return]', cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if on_close_callback:
            on_close_callback()

# ============================================================
# Main Tkinter Desktop Application
# ============================================================
class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Emotion Detection System - نظام التعرف على المشاعر")
        self.root.geometry("980x700")
        self.root.minsize(900, 650)
        self.root.configure(bg="#1a1c23")

        self.current_image_path = None
        self.camera_running = False

        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure(".", background="#1a1c23", foreground="#ffffff")
        self.style.configure("TProgressbar", thickness=14, troughcolor="#2c303c", background="#3498db")

    def _build_ui(self):
        # ── Top Header ─────────────────────────────────────────
        header_frame = tk.Frame(self.root, bg="#212530", pady=15, padx=25)
        header_frame.pack(fill=tk.X)

        title_lbl = tk.Label(
            header_frame,
            text="🎭 Facial Emotion Recognition System",
            font=("Segoe UI", 18, "bold"),
            bg="#212530",
            fg="#ffffff"
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            header_frame,
            text="Real-time DeepFace CNN & Haar Cascade Analysis | نظام الذكاء الاصطناعي لكشف المشاعر",
            font=("Segoe UI", 10),
            bg="#212530",
            fg="#9ca3af"
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # ── Action Buttons Bar ────────────────────────────────
        btn_bar = tk.Frame(self.root, bg="#1a1c23", pady=15, padx=25)
        btn_bar.pack(fill=tk.X)

        self.btn_camera = tk.Button(
            btn_bar,
            text="📷  Live Camera (الكاميرا المباشرة)",
            font=("Segoe UI", 11, "bold"),
            bg="#3b82f6",
            fg="#ffffff",
            activebackground="#2563eb",
            activeforeground="#ffffff",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.start_camera_mode
        )
        self.btn_camera.pack(side=tk.LEFT, padx=(0, 15))

        self.btn_upload = tk.Button(
            btn_bar,
            text="📁  Upload Image (تحميل صورة)",
            font=("Segoe UI", 11, "bold"),
            bg="#10b981",
            fg="#ffffff",
            activebackground="#059669",
            activeforeground="#ffffff",
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.upload_image_mode
        )
        self.btn_upload.pack(side=tk.LEFT)

        self.status_badge = tk.Label(
            btn_bar,
            text="Ready",
            font=("Segoe UI", 10, "bold"),
            bg="#2c303c",
            fg="#10b981",
            padx=12,
            pady=6
        )
        self.status_badge.pack(side=tk.RIGHT)

        # ── Main Content Area ─────────────────────────────────
        content_frame = tk.Frame(self.root, bg="#1a1c23", padx=25, pady=5)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left Column: Image Preview Card
        left_card = tk.Frame(content_frame, bg="#212530", relief="flat", padx=15, pady=15)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15), pady=(0, 15))

        preview_header = tk.Label(
            left_card,
            text="Image Preview / عرض الصورة",
            font=("Segoe UI", 12, "bold"),
            bg="#212530",
            fg="#e5e7eb"
        )
        preview_header.pack(anchor="w", pady=(0, 10))

        self.canvas_frame = tk.Frame(left_card, bg="#15171e", relief="solid", bd=1)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(
            self.canvas_frame,
            text="📷 Click 'Live Camera' to start real-time detection\n\nor\n\n📁 Click 'Upload Image' to analyze a photo",
            font=("Segoe UI", 11),
            bg="#15171e",
            fg="#6b7280",
            justify=tk.CENTER
        )
        self.image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Right Column: Emotion Analysis Results Card
        self.right_card = tk.Frame(content_frame, bg="#212530", width=360, padx=20, pady=15)
        self.right_card.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 15))
        self.right_card.pack_propagate(False)

        results_header = tk.Label(
            self.right_card,
            text="Analysis Results / نتائج التحليل",
            font=("Segoe UI", 12, "bold"),
            bg="#212530",
            fg="#e5e7eb"
        )
        results_header.pack(anchor="w", pady=(0, 12))

        # Dominant Emotion Box
        self.dominant_box = tk.Frame(self.right_card, bg="#2a2e3d", padx=15, pady=12, relief="flat")
        self.dominant_box.pack(fill=tk.X, pady=(0, 15))

        self.lbl_dominant_title = tk.Label(
            self.dominant_box,
            text="Dominant Emotion / الشعور السائد",
            font=("Segoe UI", 9),
            bg="#2a2e3d",
            fg="#9ca3af"
        )
        self.lbl_dominant_title.pack(anchor="w")

        self.lbl_dominant_val = tk.Label(
            self.dominant_box,
            text="None",
            font=("Segoe UI", 18, "bold"),
            bg="#2a2e3d",
            fg="#3498db"
        )
        self.lbl_dominant_val.pack(anchor="w", pady=(2, 2))

        self.lbl_dominant_arabic = tk.Label(
            self.dominant_box,
            text="--",
            font=("Segoe UI", 12),
            bg="#2a2e3d",
            fg="#e5e7eb"
        )
        self.lbl_dominant_arabic.pack(anchor="w")

        self.lbl_faces_count = tk.Label(
            self.dominant_box,
            text="Faces Detected: 0",
            font=("Segoe UI", 9),
            bg="#2a2e3d",
            fg="#9ca3af"
        )
        self.lbl_faces_count.pack(anchor="w", pady=(4, 0))

        # Breakdown Bars
        breakdown_title = tk.Label(
            self.right_card,
            text="Emotion Probability Breakdown:",
            font=("Segoe UI", 10, "bold"),
            bg="#212530",
            fg="#d1d5db"
        )
        breakdown_title.pack(anchor="w", pady=(5, 8))

        self.bar_rows = {}
        for emotion in ['happy', 'neutral', 'surprise', 'sad', 'angry', 'fear', 'disgust']:
            row_frame = tk.Frame(self.right_card, bg="#212530")
            row_frame.pack(fill=tk.X, pady=3)

            lbl_name = tk.Label(
                row_frame,
                text=f"{emotion.capitalize()} {emotion_arabic[emotion].split()[-1]}",
                font=("Segoe UI", 9),
                bg="#212530",
                fg="#e5e7eb",
                width=12,
                anchor="w"
            )
            lbl_name.pack(side=tk.LEFT)

            canvas_bar = tk.Canvas(row_frame, height=12, width=120, bg="#2c303c", highlightthickness=0)
            canvas_bar.pack(side=tk.LEFT, padx=6)

            lbl_pct = tk.Label(
                row_frame,
                text="0.0%",
                font=("Segoe UI", 9, "bold"),
                bg="#212530",
                fg="#9ca3af",
                width=7,
                anchor="e"
            )
            lbl_pct.pack(side=tk.RIGHT)

            self.bar_rows[emotion] = {
                'canvas': canvas_bar,
                'pct_lbl': lbl_pct
            }

        # ── Bottom Status Bar ─────────────────────────────────
        self.bottom_bar = tk.Label(
            self.root,
            text=" Antigravity AI Emotion Detection System | Student Project Ready",
            font=("Segoe UI", 9),
            bg="#15171e",
            fg="#6b7280",
            anchor="w",
            padx=15,
            pady=6
        )
        self.bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ============================================================
    # Mode 1: Live Camera
    # ============================================================
    def start_camera_mode(self):
        if self.camera_running:
            return

        self.camera_running = True
        self.btn_camera.configure(state=tk.DISABLED, bg="#4b5563")
        self.btn_upload.configure(state=tk.DISABLED, bg="#4b5563")
        self.status_badge.configure(text="Camera Running...", fg="#f59e0b")
        self.bottom_bar.configure(text=" Live Camera active. Press 'Q' on the camera window to stop.")

        def on_close():
            self.camera_running = False
            self.btn_camera.configure(state=tk.NORMAL, bg="#3b82f6")
            self.btn_upload.configure(state=tk.NORMAL, bg="#10b981")
            self.status_badge.configure(text="Ready", fg="#10b981")
            self.bottom_bar.configure(text=" Camera closed. Choose an option to continue.")

        cam_thread = threading.Thread(
            target=run_live_camera,
            args=(lambda: self.root.after(0, on_close),),
            daemon=True
        )
        cam_thread.start()

    # ============================================================
    # Mode 2: Upload Image
    # ============================================================
    def upload_image_mode(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image for Emotion Analysis",
            filetypes=[
                ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("All Files", "*.*")
            ]
        )
        if not file_path:
            return

        self.current_image_path = file_path
        self.status_badge.configure(text="Analyzing...", fg="#3b82f6")
        self.bottom_bar.configure(text=f" Analyzing image: {os.path.basename(file_path)}...")
        self.root.update()

        # Run analysis in a thread to keep GUI snappy
        threading.Thread(target=self._process_image, args=(file_path,), daemon=True).start()

    def _process_image(self, file_path):
        try:
            # Handle non-ASCII / Unicode paths on Windows safely
            stream = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(stream, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Could not read image file.")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )

            # Analyze emotion with DeepFace
            result = DeepFace.analyze(
                img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            dominant_emotion = result[0]['dominant_emotion']
            emotions_dict = result[0]['emotion']
            confidence = float(emotions_dict.get(dominant_emotion, 0))

            # Annotate image
            annotated = img.copy()
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    color = emotion_bgr.get(dominant_emotion, (0, 255, 0))
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

                    label = f"{dominant_emotion.upper()} ({confidence:.1f}%)"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    cv2.rectangle(annotated, (x, y - th - 12), (x + tw + 10, y), color, -1)
                    cv2.putText(annotated, label, (x + 5, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            else:
                # If Haar missed but DeepFace detected face region
                reg = result[0].get('region', {})
                rx, ry, rw, rh = reg.get('x', 0), reg.get('y', 0), reg.get('w', 0), reg.get('h', 0)
                if rw > 0 and rh > 0:
                    color = emotion_bgr.get(dominant_emotion, (0, 255, 0))
                    cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), color, 3)
                    label = f"{dominant_emotion.upper()} ({confidence:.1f}%)"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    cv2.rectangle(annotated, (rx, ry - th - 12), (rx + tw + 10, ry), color, -1)
                    cv2.putText(annotated, label, (rx + 5, ry - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Update GUI on main thread
            self.root.after(0, self._display_results, annotated, dominant_emotion, confidence, emotions_dict, len(faces))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", f"Failed to analyze image:\n{str(e)}"))
            self.root.after(0, lambda: self.status_badge.configure(text="Error", fg="#ef4444"))

    def _display_results(self, cv_img, dominant, confidence, emotions_dict, face_count):
        # Convert BGR to RGB for PIL / Tkinter
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Scale image to fit canvas frame proportionally
        self.canvas_frame.update_idletasks()
        max_w = max(self.canvas_frame.winfo_width() - 20, 480)
        max_h = max(self.canvas_frame.winfo_height() - 20, 380)

        pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.image_label.configure(image=tk_img, text="")
        self.image_label.image = tk_img  # Keep reference to prevent GC

        # Update Dominant Emotion Box
        hex_color = emotion_hex.get(dominant, "#ffffff")
        self.lbl_dominant_val.configure(text=f"{dominant.upper()} ({confidence:.1f}%)", fg=hex_color)
        self.lbl_dominant_arabic.configure(text=emotion_arabic.get(dominant, ''))
        self.lbl_faces_count.configure(text=f"Faces Detected: {face_count}")

        # Update Probability Bars
        for emotion, row in self.bar_rows.items():
            score = float(emotions_dict.get(emotion, 0.0))
            canvas = row['canvas']
            pct_lbl = row['pct_lbl']

            canvas.delete("all")
            bar_color = emotion_hex.get(emotion, "#3498db")
            filled_w = int((score / 100.0) * 120)
            if filled_w > 0:
                canvas.create_rectangle(0, 0, filled_w, 12, fill=bar_color, width=0)

            pct_lbl.configure(text=f"{score:.1f}%", fg=bar_color if score > 15 else "#9ca3af")

        self.status_badge.configure(text="Analysis Complete", fg="#10b981")
        self.bottom_bar.configure(text=f" Analysis complete: Dominant emotion is {dominant.upper()} ({confidence:.1f}%)")

# ============================================================
# Application Entry Point
# ============================================================
def main():
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
