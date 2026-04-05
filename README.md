# AI-Powered-Face-Mask-Surveillance-System
An end-to-end computer vision solution designed for automated face mask compliance monitoring in hospital environments. This system leverages YOLOv8, ByteTrack, and custom business logic to ensure public health safety with high precision and hardware efficiency.

## 🚀 Key Features

* **Real-time Hybrid Detection**: Classifies individuals into three categories: `with_mask`, `without_mask`, and `mask_weared_incorrect`.
* **Intelligent Violation Logic**:
    * **Persistence Check**: Only logs a violation if a person remains unmasked for more than **10 seconds** to avoid false alarms from temporary adjustments.
    * **De-duplication**: Prevents storage bloat by logging a single evidence photo per person every **2 hours** using **Object Tracking (Unique ID assignment)**.
* **Advanced Data Engineering**:
    * **Selective Cropping & Upsampling**: Solved extreme class imbalance by isolating and duplicating minority class features (incorrectly worn masks) without diluting the majority class.
* **Hardware Optimized**: Ready for deployment on hospital edge servers via **OpenVINO** and **INT8 Quantization**, achieving ~30 FPS on standard CPUs.

## 📊 Technical Performance

| Metric | Value |
| :--- | :--- |
| **mAP@50 (Overall)** | **0.7851** |
| **with_mask mAP@50** | **0.939** |
| **without_mask mAP@50** | **0.829** |
| **Inference Speed** | **2.2ms - 5.9ms** (GPU) |

> **Note**: The `mask_weared_incorrect` class was specifically improved using a custom **Selective Cropping Pipeline**, boosting its recall and making the model reliable for catching "nose-out" or "chin-only" mask-wearing cases.

## 🏗️ System Architecture

1.  **Input**: High-definition RTSP streams from hospital hall cameras or local webcams.
2.  **Tracking**: ByteTrack assigns a unique ID to every person entering the frame to monitor individual behavior over time.
3.  **Inference**: YOLOv8 (optimized with INT8) detects mask status on each tracked object.
4.  **Logic Controller**: 
    * Checks if `violation_duration > 10s`.
    * Verifies if `last_save_time > 2h` for that specific ID.
    * Saves high-res face crop to `violations_log/`.
5.  **Output**: Real-time visual monitor and archived proof of violations.

## 🛠️ Installation & Usage

### 1. Environment Setup
Ensure you have **Python 3.10 - 3.12** installed. It is highly recommended to use a virtual environment to avoid conflicts:


# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

### 2. Install Dependencies
Install all necessary libraries using the provided requirements file to ensure version compatibility:
pip install -r requirements.txt


### 3. File Organization
Place the following files in your root directory to ensure the demo script can correctly locate the model:
* `demo.ipynb`: The surveillance logic notebook.
* `best_model.pt`: The trained YOLOv8 model weights.
* `hospital_violations/`: Folder where violation evidence will be saved (automatically created).

### 4. Run Surveillance Demo
1. Open `demo.ipynb` in your preferred editor (Jupyter or VS Code).
2. Verify the model loading line: `model = YOLO('best_model.pt')`.
3. Execute all cells to start real-time monitoring via your webcam.
