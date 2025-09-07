import cv2
import numpy as np
import onnxruntime as ort

class EmotionModel:
    def __init__(self, model_path="models/emotion-ferplus-8.onnx", use_gpu=False):
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.emotions = ["neutral", "happiness", "surprise", "sadness",
                         "anger", "disgust", "fear", "contempt"]

    def preprocess(self, face_img):
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Resize to 64x64
        resized = cv2.resize(gray, (64, 64))
        # Normalize to [-1,1]
        tensor = resized.astype("float32") / 255.0
        tensor = (tensor - 0.5) * 2.0
        # Add batch and channel dimensions (1,1,64,64)
        tensor = np.expand_dims(tensor, axis=0)
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def predict(self, face_img):
        input_tensor = self.preprocess(face_img)
        # Debug: check input tensor
        # print("Tensor shape:", input_tensor.shape, "Min/Max:", input_tensor.min(), input_tensor.max())
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        idx = int(np.argmax(outputs))
        confidence = float(np.max(outputs))
        return self.emotions[idx], confidence
