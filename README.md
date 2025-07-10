# Player Re-Identification and Tracking

## Objective
This project processes a 15-second video (`15sec_input_720p.mp4`) to detect and track players using object detection and re-identification. Players who leave and re-enter the frame are assigned the same ID, simulating real-time tracking.

---

## Setup Instructions

### 1. Clone or Download the Project
Place all provided files (`main.py`, `requirements.txt`, `best.pt`, `15sec_input_720p.mp4`) in the same directory.

### 2. Install Dependencies
Open a terminal in the project directory and run:
```bash
pip install -r requirements.txt
```
This will install all required Python packages, including:
- opencv-python
- numpy
- ultralytics
- torch
- Pillow
- torchreid
- torchvision
- scikit-learn
- gdown
- lap

> **Note:**
> - Python 3.8+ is recommended.
> - CUDA-enabled GPU is recommended for faster processing, but CPU will also work (slower).

### 3. Run the Code
In the same terminal, run:
```bash
python main.py
```

- The script will process `15sec_input_720p.mp4` using the YOLO model (`best.pt`).
- Output video with tracked and re-identified players will be saved as:
  - `tracked_video_reid_final.mp4`
- Debug information will be printed in the terminal.

---

## How It Works
- **Detection:** Uses YOLO to detect players in each frame.
- **Re-Identification:** Uses `torchreid` to extract features and match players who re-enter the frame.
- **Tracking:** Maintains unique IDs for each player, even if they leave and re-enter.
- **Output:** Annotated video with bounding boxes and player IDs.

---

## Troubleshooting
- If you see errors about missing packages, re-run `pip install -r requirements.txt`.
- If the output video is empty or corrupted, ensure all dependencies are installed and try running again.
- For codec issues, ensure you have the latest version of OpenCV and FFMPEG installed.

---

## Contact
For any issues or questions, please raise an issue or contact the project maintainer.

---

## Sources Used for the Assignment

- ChatGPT (learned about YOLO algorithm)
- Google (used to resolve problems encountered during the project)
- YouTube: https://www.youtube.com/watch?v=IG1h0zZC1Nk 