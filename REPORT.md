# Player Re-Identification and Tracking: Brief Report

**Author:** Vishal Yadav

## 1. Approach and Methodology
- **Objective:**
  - Detect and track each player in a 15-second football video, ensuring that players who leave and re-enter the frame are assigned the same identity.
- **Pipeline:**
  1. **Object Detection:**
     - Used a custom YOLO model (`best.pt`) to detect players, referees, and goalkeepers in each frame.
  2. **Feature Extraction (Re-ID):**
     - Used `torchreid` (OSNet) to extract appearance features from each detected player.
  3. **ID Assignment:**
     - Assigned a unique global ID to each new player.
     - When a player re-entered, compared their features to a gallery of previously seen players using cosine similarity, and reassigned the same ID if matched.
  4. **Output:**
     - Drew bounding boxes and IDs on each frame and saved the annotated video.

## 2. Techniques Tried and Outcomes
- **YOLO Object Detection:**
  - Provided robust detection of players and other entities in each frame.
- **Re-Identification with OSNet:**
  - Enabled matching of players who left and re-entered the frame, maintaining consistent IDs.
- **Cosine Similarity for Matching:**
  - Used a threshold to decide if a re-entering player matches a previously seen player.
- **Frame-by-Frame Processing:**
  - Simulated real-time tracking and re-identification.

**Outcomes:**
- Successfully generated an output video with consistent player IDs, even after players left and re-entered the frame.
- Debugging and logging confirmed that frames were written correctly and IDs were maintained.

## 3. Challenges Encountered
- **Dependency Issues:**
  - Some required packages (like `gdown`, `lap`) were not initially installed, causing runtime errors.
- **VideoWriter Codec Problems:**
  - The output video was sometimes corrupted due to codec or frame size mismatches. Switching codecs and ensuring frame size consistency resolved this.
- **Model Download Delays:**
  - The first run required downloading pretrained weights, which took extra time.
- **Re-ID Threshold Tuning:**
  - Setting the right similarity threshold was important to avoid ID switches or duplicates.

## 4. If Incomplete: Next Steps
- **Current Status:**
  - All core requirements are met: detection, tracking, re-identification, and output video.
- **Possible Improvements:**
  - Fine-tune the re-ID threshold for even more robust ID assignment.
  - Add more advanced tracking (e.g., Kalman filter, motion models) for smoother ID continuity.
  - Improve visualization (e.g., color-coding IDs, showing track history).
  - Test on longer or more complex videos for further validation.

---

**Conclusion:**
The solution meets the assignment objectives, providing real-time player re-identification and tracking in a single video feed. All major challenges were addressed, and the code is ready for further extension or deployment. 