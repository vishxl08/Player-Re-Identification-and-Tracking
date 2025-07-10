"""
Player Re-Identification and Tracking
Author: Vishal Yadav

- Detects and tracks players in a video using YOLO and OSNet (torchreid).
- Maintains consistent IDs for players, even if they leave and re-enter the frame.
- Outputs an annotated video with bounding boxes and IDs.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchreid
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm

# ------------------- CONFIGURATION -------------------
VIDEO_PATH = r"D:/test/15sec_input_720p.mp4"
MODEL_PATH = r"D:/test/best.pt"
OUTPUT_DIR = r"D:/test"
OUTPUT_VIDEO_PATH = r"D:/test/tracked_video_reid_final.mp4"
SIMILARITY_THRESHOLD = 0.7
DEBUG = False  # Set True for debug prints
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- MODEL SETUP -------------------
def load_reid_model() -> torch.nn.Module:
    """Load the OSNet re-identification model."""
    model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
    model.to(device)
    model.eval()
    return model

reid_model = load_reid_model()

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ------------------- FEATURE EXTRACTION -------------------
def extract_features(image_crop: np.ndarray) -> Optional[np.ndarray]:
    """Extract appearance features from a cropped player image."""
    try:
        img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = reid_model(tensor)
        return features.cpu().numpy().flatten()
    except Exception as e:
        if DEBUG:
            print(f"Feature extraction failed: {e}")
        return None

# ------------------- TRACKING LOGIC -------------------
class TrackManager:
    """Manages active and inactive tracks for player re-identification."""
    def __init__(self, similarity_threshold: float = 0.7):
        self.global_id_counter = 0
        self.active_tracks: Dict[int, Dict] = {}
        self.inactive_gallery: List[Dict] = []
        self.similarity_threshold = similarity_threshold

    def match_in_gallery(self, features: np.ndarray, used_global_ids: Set[int]) -> Optional[int]:
        """Find a matching global ID in the inactive gallery based on feature similarity."""
        filtered = [g for g in self.inactive_gallery if g['global_id'] not in used_global_ids]
        if not filtered:
            return None
        gallery_features = [g['features'] for g in filtered]
        gallery_ids = [g['global_id'] for g in filtered]
        sims = cosine_similarity([features], gallery_features)[0]
        best_idx = np.argmax(sims)
        if sims[best_idx] > self.similarity_threshold:
            return gallery_ids[best_idx]
        return None

    def assign_global_id(self, track_id: int, bbox: Tuple[int, int, int, int], frame: np.ndarray, used_global_ids: Set[int]) -> Optional[int]:
        """Assign or re-assign a global ID to a detected player."""
        if track_id in self.active_tracks:
            global_id = self.active_tracks[track_id]['global_id']
            if global_id not in used_global_ids:
                return global_id
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        features = extract_features(crop)
        if features is None:
            return None
        matched_global_id = self.match_in_gallery(features, used_global_ids)
        if matched_global_id is not None:
            global_id = matched_global_id
        else:
            self.global_id_counter += 1
            global_id = self.global_id_counter
            self.inactive_gallery.append({'global_id': global_id, 'features': features})
        self.active_tracks[track_id] = {'global_id': global_id, 'features': features}
        used_global_ids.add(global_id)
        return global_id

    def retire_lost_tracks(self, current_track_ids: List[int]):
        """Move lost tracks to the inactive gallery."""
        lost_ids = set(self.active_tracks.keys()) - set(current_track_ids)
        for tid in lost_ids:
            if 'features' in self.active_tracks[tid]:
                self.inactive_gallery.append({
                    'global_id': self.active_tracks[tid]['global_id'],
                    'features': self.active_tracks[tid]['features']
                })
        for tid in lost_ids:
            del self.active_tracks[tid]

# ------------------- DRAWING -------------------
def draw_frame(frame: np.ndarray, results, class_names: Dict, track_manager: TrackManager) -> np.ndarray:
    """Draw bounding boxes and IDs on the frame."""
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color_map = {
        'player': (255, 255, 255),
        'referee': (0, 215, 255),
        'goalkeeper': (0, 0, 255)
    }
    used_global_ids = set()
    current_frame_track_ids = []
    if results.boxes.id is not None:
        boxes = results.boxes.data.cpu().numpy()
        for *xyxy, track_id, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            track_id = int(track_id)
            cls_id = int(cls_id)
            label = class_names.get(cls_id, f"class{cls_id}")
            if label == 'ball':
                continue
            current_frame_track_ids.append(track_id)
            global_id = track_manager.assign_global_id(track_id, (x1, y1, x2, y2), frame, used_global_ids)
            color = color_map.get(label, (128, 128, 128))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            text = f"ID: {global_id}" if global_id is not None else "ID: ?"
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    track_manager.retire_lost_tracks(current_frame_track_ids)
    return annotated

# ------------------- MAIN PROCESSING LOOP -------------------
def run_tracking(video_path: str, model_path: str, output_dir: str, output_video_path: str, similarity_threshold: float = 0.7):
    """Main function to process the video and save the output with tracked IDs."""
    model = YOLO(model_path)
    class_names = model.names
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open input video: {video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out_writer.isOpened():
        print(f"Failed to open VideoWriter for: {output_video_path}")
        return
    frame_idx = 0
    track_manager = TrackManager(similarity_threshold)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(source=frame, persist=True, conf=0.4, verbose=False)[0]
            output_frame = draw_frame(frame, results, class_names, track_manager)
            # Ensure frame size matches
            if output_frame.shape[1] != width or output_frame.shape[0] != height:
                if DEBUG:
                    print(f"Resizing frame from {output_frame.shape[1]}x{output_frame.shape[0]} to {width}x{height}")
                output_frame = cv2.resize(output_frame, (width, height))
            out_writer.write(output_frame)
            frame_idx += 1
            pbar.update(1)
    cap.release()
    out_writer.release()
    print(f"Processed {frame_idx} frames. Video saved to {output_video_path}")

# ------------------- ENTRY POINT -------------------
if __name__ == "__main__":
    run_tracking(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR, OUTPUT_VIDEO_PATH, SIMILARITY_THRESHOLD)