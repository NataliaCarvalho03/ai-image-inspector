import sys
import os

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    PyInstaller creates a temp folder and stores path in _MEIPASS.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_onnx_providers():
    """ 
    Automatically detect the presence of GPU (CUDA) or CoreML via ONNX Runtime,
    falling back to CPU if no compatible GPU is found.
    """
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    providers = []
    
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
    if 'CoreMLExecutionProvider' in available_providers:
        providers.append('CoreMLExecutionProvider')
    
    # Fallback to CPU
    providers.append('CPUExecutionProvider')
    return providers
