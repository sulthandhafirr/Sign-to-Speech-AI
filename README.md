# Sign to Speech AI with YOLOv11n

This project provides an AI-based solution to recognize hand signs using the **YOLOv11n** object detection model and convert them into real-time speech using **Google Text-to-Speech (gTTS)**. It is designed to help bridge communication between sign language users and non-signers through camera-based gesture detection.

## Dataset

Before training, download and extract the dataset:

**📁 dataset4.zip**

This dataset is included in the repository and contains labeled image data representing sign language gestures.


## Model Training

Training is handled using the notebook: `YOLOv11n_train.ipynb`

You can run this notebook in Google Colab or locally via Jupyter Notebook.

### Training Steps:
1. Open `YOLOv11n_train.ipynb`
2. Run the notebook step-by-step
3. Ensure the dataset path is correctly set to your extracted `signdataset/` folder
4. The trained model will be exported to ONNX format (e.g., `yolov11n_sign.onnx`)


## How to Run `app.py` Locally

This application runs in a local Python environment using your webcam. It detects hand gestures and converts them into speech output.

### 1. Download Files

Ensure the following files are in the same folder:
- `app.py`
- `yolov11n_sign.onnx`
- (optional) `requirements.txt`

### 2. Install Dependencies

Make sure Python 3.8+ is installed. Then run:

```
pip install -r requirements.txt
```

Or install manually:

```
pip install opencv-python numpy gTTS ultralytics
```

# 3. Run the Application
Open a terminal, navigate to the project folder, and run:

bash
Copy
Edit
python app.py
The webcam will open, and the model will detect hand signs in real time. Detected gestures will be translated to text and spoken using gTTS.

# Technologies Used
YOLOv11n – Lightweight object detection model (Ultralytics)

OpenCV – Webcam video feed & image processing

NumPy – Array and matrix operations

gTTS (Google Text-to-Speech) – Converts recognized gestures to spoken output

ONNX – For running exported PyTorch models

# Credits
Sign Language Datasets: Public sources (Google Images, Roboflow, etc.)

Object Detection: YOLOv11n (Ultralytics community version)

Model Conversion: PyTorch → ONNX

Audio Output: gTTS

Real-time Detection: OpenCV + Python
