# AI Image Inspector

A cross-platform desktop application built with Python and CustomTkinter that combines classical CNN-based object detection with modern Vision-Language Model (VLM) capabilities.

## 🚀 Features

- **Object Detection**: Rapidly identify objects in images using YOLO-based CNN models via ONNX Runtime and OpenCV.
- **Image Description**: Generate detailed natural language descriptions of images using the Moondream2 Vision-Language Model.
- **Cross-Platform**: Sleek, modern GUI that works on Linux, Windows, and macOS.
- **Hardware Acceleration**: Automatic detection and utilization of CUDA (NVIDIA) or CoreML (Apple Silicon/macOS) for faster inference, with seamless CPU fallback.
- **Asynchronous Execution**: Model loading and inference are performed in background threads to keep the UI responsive.

## 🛠️ Technology Stack

- **GUI**: [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- **Object Detection**: OpenCV DNN / ONNX Runtime
- **VLM Inference**: Moondream2 (via ONNX Runtime)
- **Image Processing**: Pillow (PIL), NumPy, OpenCV
- **Tokenization**: HuggingFace Tokenizers

## 📋 Prerequisites

- Python 3.8+
- (Optional) NVIDIA GPU for CUDA acceleration

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-image-inspector
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Model Setup:**
   Ensure the following model files are present in the `models/` directory:
   - `models/yolo.onnx` (for object detection)
   - `models/moondream/tokenizer.json`
   - `models/moondream/onnx/vision_encoder_q4.onnx`
   - `models/moondream/onnx/decoder_model_merged_bnb4.onnx`
   - `models/moondream/onnx/model_bnb4.onnx` (used for embedding extraction)

## 🏃 Usage

Launch the application by running:
```bash
python main.py
```

1. **Load Image**: Click the "Load Image" button to select a file.
2. **Object Detection**: Switch to the "Object Detection" tab and click "Run Object Detection".
3. **VLM Description**: Switch to the "VLM Description" tab, enter an optional prompt, and click "Generate Description".

## 📁 Project Structure

```text
ai-image-inspector/
├── main.py              # Application entry point
├── src/
│   ├── ui.py            # CustomTkinter GUI implementation
│   ├── engine_cnn.py    # CNN detection logic (YOLO)
│   ├── engine_vlm.py    # VLM inference logic (Moondream2)
│   ├── utils.py         # Path and hardware helper functions
├── models/              # ONNX models and tokenizers (not included in git)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## ⚖️ License

[MIT License](LICENSE) (or specify your preferred license)
