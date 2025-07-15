# Sign to Speech AI with YOLOv11n

This project provides an AI-based solution to recognize hand signs using the **YOLOv11n** object detection model and convert them into real-time speech using **Google Text-to-Speech (gTTS)**. It is designed to help bridge communication between sign language users and non-signers through camera-based gesture detection.

# Dataset

Before training, download and extract the dataset:

**📁 [dataset4.zip](https://drive.google.com/file/d/1MbMCORWnfd-g7gx3AAhNqBOff7CSg3je/view?usp=sharing)**

This dataset is included in the repository and contains labeled image data representing sign language gestures.


# Model Training

Training is handled using the notebook: `SigntoSpeechAI.ipynb`

You can run this notebook in Google Colab or locally via Jupyter Notebook.

### Training Steps:
1. Open `SigntoSpeechAI.ipynb`
2. Set the correct dataset path
3. Run the notebook step-by-step
5. Export to ONNX or use `.pt` for local inference

# How to Run `app.py` Locally

This application runs in a local Python environment using your webcam. It detects hand gestures and converts them into speech output.

### 1. Download Files

Ensure the following files are in the same folder:
- `app.py`
- `best.pt`
- (optional) `requirements.txt`
- bisindow.jpg

### 2. Install Dependencies

Make sure Python 3.8+ is installed. Then run:

```
pip install -r requirements.txt
```

Or install manually:

```
pip install opencv-python numpy gTTS playsound==1.2.2 ultralytics
```

### 3. Run the Application
Open a terminal, navigate to the project folder, and run:

```bash
python app.py
```
The webcam will open, and the model will detect hand signs in real time. Detected gestures will be translated to text and spoken using gTTS.

# Instructions (in-app)
- [Q] Quit
- [R] Reset
- HALLO once → space
- HALLO twice → speak text

# Credits
- Dataset sources: Roboflow
- Object Detection: YOLOv11n by Ultralytics
- Model Export: PyTorch → ONNX
- Real-time Detection: OpenCV
- Audio Output: gTTS, Playsound
- Development: Python, VS Code
- Model Training: Google Colab

---
Developed by: Sulthan Dhafir Rafief
