# 🎭 Facial Emotion Recognition System
### نظام التعرف على المشاعر بالذكاء الاصطناعي

A real-time facial emotion detection desktop application built with **Python**, **OpenCV**, **DeepFace**, and **Tkinter**. Supports both live webcam detection and static image analysis, with a bilingual (English/Arabic) dark-themed interface.

---

## 📋 Requirements

| Requirement | Version |
|---|---|
| Python | 3.9 – 3.11 |
| Operating System | Windows 10 / 11 |
| Webcam | Required for Live Camera mode |
| Internet | Required **first run only** (to download model weights) |

---

## ⚡ Quick Setup (Step by Step)

### Step 1 — Clone or Download the Project

Place the project folder anywhere on your computer. The folder should contain:
```
emotion_project/
├── emotion_detection.py
├── requirements.txt
├── run.bat
└── haarcascade_frontalface_default.xml
```

### Step 2 — Open a Terminal in the Project Folder

Right-click inside the project folder → **"Open in Terminal"** (or PowerShell)

### Step 3 — Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4 — Install Dependencies

```bash
.venv\Scripts\pip install -r requirements.txt
```

> ⚠️ **Note:** The first install downloads TensorFlow (~350 MB) and OpenCV (~40 MB). Make sure you have a stable internet connection. If it times out, just run the same command again — pip will resume from cache.

### Step 5 — Run the Application

**Option A — Double-click** `run.bat` in the project folder

**Option B — Terminal:**
```bash
.venv\Scripts\python emotion_detection.py
```

---

## 🚀 First Run Note

On the **very first launch**, DeepFace will automatically download the emotion model weights (~6 MB):
- `haarcascade_frontalface_default.xml`
- `facial_expression_model_weights.h5`

These are saved to `C:\Users\<YourName>\.deepface\weights\` and are **only downloaded once**.

---

## 🖥️ How to Use the Application

### Main Window

| Element | Description |
|---|---|
| **📷 Live Camera** button | Starts real-time webcam emotion detection |
| **📁 Upload Image** button | Opens a file picker to analyze a saved photo |
| **Image Preview** (left panel) | Displays the annotated image/frame with face bounding boxes |
| **Analysis Results** (right panel) | Shows dominant emotion + Arabic translation + 7-emotion probability bars |
| **Status Badge** | Shows current state: Ready / Analyzing / Camera Running / Complete |

### Live Camera Mode

1. Click **📷 Live Camera**
2. A new OpenCV window opens with your webcam feed
3. Faces are detected automatically with colored bounding boxes
4. Emotion label and confidence % shown above each face
5. Probability bars displayed beside each detected face
6. Press **`Q`** on the camera window to close it and return to the main GUI

### Upload Image Mode

1. Click **📁 Upload Image**
2. A file picker opens — select any `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` file
3. The system analyzes the image (may take 2–5 seconds)
4. The annotated image is shown in the preview panel
5. Full emotion breakdown is displayed on the right panel

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | 4.10.0.84 | Face detection & video capture |
| `deepface` | latest | CNN-based emotion recognition |
| `tensorflow` | 2.21.x | Deep learning backend for DeepFace |
| `tf-keras` | 2.21.x | Keras compatibility layer |
| `numpy` | latest | Image array processing |
| `pillow` | latest | Image rendering in Tkinter GUI |

---

## 🗂️ Project Structure

```
emotion_project/
│
├── emotion_detection.py              ← Main application file
├── requirements.txt                  ← Python dependencies list
├── run.bat                           ← One-click Windows launcher
├── haarcascade_frontalface_default.xml  ← Face detection classifier
│
└── .venv/                            ← Virtual environment (auto-created)
    └── Scripts/
        ├── python.exe
        └── pip.exe
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: deepface` | Run `.venv\Scripts\pip install -r requirements.txt` again |
| `Error: Cannot open camera!` | Make sure no other app (Teams, Zoom, etc.) is using the webcam |
| App window doesn't appear | Check your Windows taskbar — the window may be loading in background |
| Image shows broken characters | Use `.jpg` or `.png` files with simple ASCII file names |
| First run is very slow | Normal — DeepFace downloads model weights (~6 MB) once |
| `pip` install times out | Re-run the install command — pip resumes from cache |

---

## 👨‍💻 Technologies Used

- **Python 3.11** — Core programming language
- **OpenCV** — Real-time computer vision, face detection via Haar Cascades
- **DeepFace** — Pre-trained CNN model (FER-2013 dataset) for emotion classification
- **TensorFlow / tf-Keras** — Deep learning backend
- **Tkinter** — Native Python GUI framework
- **Pillow (PIL)** — Image display and conversion in GUI
- **Threading** — Non-blocking camera and analysis operations

---

## 📄 License

This project is developed for academic/educational purposes.
