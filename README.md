# Face Recognition Attendance System

This repository contains a face-recognition-based attendance system built with Python. The project provides scripts to train face encodings, recognize faces from a camera or images, and maintain attendance logs.

## Features

- Detect and recognize faces using `dlib`/`face_recognition` libraries.
- Record attendance to CSV files.
- Web UI and scripts for adding users and running recognition.

## Repository structure

- `FACE.py`, `final.py`, `1.py` - main scripts for face detection/recognition and attendance logging.
- `static/face_encodings.pkl` - Serialized face encodings for known users.
- `static/faces/` - Directory containing subfolders for each user with their images.
- `templates/` - HTML templates for the web UI (e.g., `home.html`, `adduser.html`).
- `haarcascade_frontalface_alt.xml` - Haar cascade for face detection fallback.
- `dlib` wheel files - Pre-built `dlib` wheels included for certain Python versions.

## Setup

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Install dependencies. If a `requirements.txt` is not present, install the common packages:

```bash
pip install flask face_recognition dlib opencv-python numpy pandas
```

Note: Installing `dlib` can be platform-specific. Pre-built wheels are included for some Windows Python versions; use the matching wheel or build from source.

## Running

1. To add a new user, place images in `static/faces/<USER_NAME>/` and run the script that creates encodings or use the web UI `adduser.html` if available.
2. Generate/update `static/face_encodings.pkl` by running the encoding script (look for functions in `data.py` or `FACE.py`).
3. Run the recognition/attendance script:

```bash
python FACE.py
# or
python final.py
```

4. Check the generated attendance CSV (e.g., `Attendance/Attendance-05_24_25.csv`) for logs.

## Usage

- Use the web interface to manage users and start recognition, or run the Python scripts directly.
- Ensure the camera is accessible and permissions are granted.

## Notes

- If face recognition fails, verify the models and encodings are present in `static/` and that the faces directory is structured correctly.
- The included `dlib` wheels are for Windows and specific Python versions; building `dlib` from source may be required on other platforms.

## Ultralytics YOLO (optional)

- Some scripts in this workspace may use Ultralytics YOLO for object/face detection (look for imports from `ultralytics` in `project_yolo_trainer`). If you intend to use Ultralytics YOLO with this face recognition project, prefer using a `final` weights file exported from training (commonly named `best.pt` or `final.pt`) and load it via:

```python
from ultralytics import YOLO
model = YOLO('path/to/final.pt')
```

- Ultralytics requires a compatible environment; if you use it, install the package with:

```bash
pip install ultralytics
```

- Note: Ultralytics models are optional for this project; the primary face recognition flow uses `face_recognition` and `dlib`.

## `videoly.py` (image-only utility — no live video)

- The repository includes or can include a small utility named `videoly.py` intended for processing video files or image sequences using detection/recognition code, but configured to run without live camera capture. This utility is compatible with Python 3.10.

- Purpose: run face detection/recognition on a single image or on frames extracted from a video file without opening a live camera stream. Useful for batch-processing recordings or testing on saved images.

- Example usage (image input):

```bash
python videoly.py --input images/sample.jpg --output results/output.jpg --model static/face_encodings.pkl
```

- Example usage (video -> process frames without display):

```bash
python videoly.py --input videos/lecture.mp4 --output_dir results/frames --no-display
```

- Implementation notes for `videoly.py`:
  - Should accept command-line arguments: `--input`, `--output` or `--output_dir`, `--model`, and `--no-display`.
  - Must not attempt to open a camera device; use `cv2.VideoCapture` only on file paths.
  - Target Python version: 3.10 (ensure any f-strings or syntax are compatible).

- Dependencies (if not already installed):

```bash
pip install opencv-python face_recognition numpy
```

## License

Add a license if you plan to distribute this project publicly.
