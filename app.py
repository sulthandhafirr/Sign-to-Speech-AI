import cv2
from ultralytics import YOLO
import time
import numpy as np
from gtts import gTTS
from playsound import playsound
import os
import uuid
import textwrap

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

detected_text = ""
prev_label = ""
current_label = ""
label_start_time = 0
last_flash_time = 0
show_flash = False

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
line_height = 40
max_text_area_width = 1000

frame_width, frame_height = 640, 480

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    results = model.predict(source=frame, conf=0.5, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    annotated_frame = frame.copy()

    if boxes is not None:
        for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = names[int(cls_id.item())]
            conf_text = f"{label} {conf:.2f}" if label != "HALLO" else ""
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if conf_text:
                cv2.putText(annotated_frame, conf_text, (x1, y1 - 10),
                            font, 0.8, (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        detected_text = ""

    if boxes and names:
        top_result = boxes.conf.argmax().item()
        label_id = int(boxes.cls[top_result].item())
        label = names[label_id]

        if label != current_label:
            current_label = label
            label_start_time = time.time()
        elif time.time() - label_start_time >= 0.5 and label != prev_label:
            detected_text += " " if label == "HALLO" else label
            prev_label = label
            last_flash_time = time.time()
            show_flash = True
    else:
        current_label = ""
        label_start_time = time.time()
        prev_label = ""

    if "  " in detected_text:
        speak_text = detected_text.strip()
        print(f"Membaca: {speak_text}")
        tts = gTTS(text=speak_text, lang='id')
        filename = f"temp_{uuid.uuid4().hex}.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
        detected_text = ""

    if show_flash and time.time() - last_flash_time < 0.1:
        overlay = annotated_frame.copy()
        white_overlay = (255 * np.ones_like(annotated_frame)).astype('uint8')
        alpha = 0.5
        cv2.addWeighted(white_overlay, alpha, overlay, 1 - alpha, 0, overlay)
        annotated_frame = overlay
    else:
        show_flash = False

    right_panel = np.ones((480, 640, 3), dtype=np.uint8) * 255
    ref_path = "bisindow.jpg"
    if os.path.exists(ref_path):
        ref_img = cv2.imread(ref_path)
        size = 480
        ref_img = cv2.resize(ref_img, (size, size))
        x_offset = (640 - size) // 2
        y_offset = (480 - size) // 2
        right_panel[y_offset:y_offset+size, x_offset:x_offset+size] = ref_img
    else:
        cv2.putText(right_panel, "Error not found", (20, 60), font, 0.7, (0, 0, 255), 2)

    top_combined = np.hstack((annotated_frame, right_panel))

    cursor_char = "|" if int(time.time() * 2) % 2 == 0 else " "
    full_text = (detected_text.strip() + cursor_char).strip()

    wrapped_lines = textwrap.wrap(full_text, width=60)
    max_lines = 3
    displayed_lines = wrapped_lines[-max_lines:]

    bottom_panel = np.ones((200, 1280, 3), dtype=np.uint8) * 255

    cv2.putText(bottom_panel, "Output:", (20, 40), font, 1, (0, 0, 0), 2)
    for i, line in enumerate(displayed_lines):
        y = 80 + i * line_height
        cv2.putText(bottom_panel, line, (20, y), font, 0.9, (50, 50, 50), 2)

    cv2.putText(bottom_panel, "[Q] Quit", (1050, 40), font, 0.8, (0, 100, 0), 2)
    cv2.putText(bottom_panel, "[R] Reset", (1050, 80), font, 0.8, (0, 0, 255), 2)

    full_view = np.vstack((top_combined, bottom_panel))
    cv2.imshow("Sign Language Detection Interface", full_view)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()