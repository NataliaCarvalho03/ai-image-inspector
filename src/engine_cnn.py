import cv2
import numpy as np
import os
from src.utils import resource_path

class CNNEngine:
    def __init__(self, model_path="models/yolo.onnx"):
        self.model_path = resource_path(model_path)
        self.net = None
        
        if not os.path.exists(self.model_path):
            print(f"Warning: CNN Model file not found at {self.model_path}")
            return
            
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            build_info = cv2.getBuildInformation()
            has_cuda_support = "CUDA: YES" in build_info
            # Hardware Management: attempt to use CUDA if available, fallback automatically
            if has_cuda_support:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using CUDA")
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print("Using CPU")
                
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            
    def is_loaded(self):
        return self.net is not None

    def detect(self, image, confidence_threshold=0.5):
        if not self.is_loaded():
            return f"Error: CNN model not found at {self.model_path}. Please place your .onnx model there."
            
        # Pre-processing (blob from image)
        # Note: Size may vary depending on the specific model (e.g. YOLOv8 uses 640x640)
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Network execution
        try:
            outputs = self.net.forward()
        except Exception as e:
            print(e)
            return f"Error during inference: {e}"
        
        # Post-processing (Non-Maximum Suppression - NMS)
        # This is a generic handling shape commonly seen in YOLO ONNX exports
        if len(outputs.shape) == 3:
            outputs = outputs[0]
            outputs = outputs.T
            
        boxes = []
        confidences = []
        class_ids = []
        
        for row in outputs:
            classes_scores = row[4:]
            if len(classes_scores) > 0:
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                score = classes_scores[class_id]
                
                if score > confidence_threshold:
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                    left = int(x - w / 2)
                    top = int(y - h / 2)
                    
                    boxes.append([left, top, int(w), int(h)])
                    confidences.append(float(score))
                    class_ids.append(class_id)
                
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "class_id": class_ids[i], 
                    "confidence": confidences[i], 
                    "box": boxes[i]
                })
                
        return results
