# Team name - Quantum Coders 
```
Members - Piyush Lamba (Leader) , Abhishek, AjayPal Singh, Rajveer Kaur
```
# Face Recognition Attendance System

A real-time **face recognition–based attendance system** built in Python using **OpenCV** and **face_recognition**.  
It detects faces from your webcam feed, recognizes known people, and automatically logs their attendance (Name, Date, Time) into a CSV file.

---

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/attendance-mern.git
cd attendance-mern
```

---

### 2. Core Libraries Used
Here’s the list you must install/import for this Python script to run:

```python
import os
import time
import csv
import cv2                 # pip install opencv-python
import numpy as np         # pip install numpy
import face_recognition    # pip install face-recognition
from datetime import datetime, date
```

---

### 3. Install All Dependencies
Run this in your terminal:

```bash
pip install opencv-python numpy face-recognition opencv-contrib-python
```

---

### Note on dlib Installation
`face_recognition` depends on **dlib**.  
If you face issues (especially on Windows or Python 3.12), install CMake and the correct dlib wheel:

```bash
pip install cmake
pip install dlib-19.24.2-cp312-cp312-win_amd64.whl
```

> Download the correct `.whl` for your system from:  
> [https://pypi.org/project/dlib/](https://pypi.org/project/dlib/)  
> or [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#dlib)

---

## Usage

1. Create a folder named `images/` and add subfolders or images of known people:  
   ```
   images/
    ├── Alice/
    │   └── alice.jpg
    ├── Bob/
    │   └── bob.png
   ```
2. Ensure the following files are in the project directory:
   - `deploy.prototxt.txt`
   - `res10_300x300_ssd_iter_140000.caffemodel`
3. Run the project:
   ```bash
   python attendace.py
   ```
4. A webcam window will open and start detecting/recognizing faces.
5. Press **`q`** to quit.

---

## Output Example

Generated file: `attendance_optimiiizenhjk.csv`

| Name  | Date       | Time     |
|--------|------------|----------|
| Alice  | 2025-10-04 | 09:22:31 |
| Bob    | 2025-10-04 | 09:25:47 |

---

## 📚 Credits

- **OpenCV DNN Face Detector** – [OpenCV Model Zoo](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)  
- **face_recognition** Library – [https://github.com/ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)