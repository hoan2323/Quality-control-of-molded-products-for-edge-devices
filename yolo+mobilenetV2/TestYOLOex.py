import cv2
import numpy as np
import onnxruntime as ort

class YOLODetector:
    def __init__(self, model_path, conf_threshold=0.75, iou_threshold=0.5):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = ['Casting']
        self.input_size = 320
        
    def preprocess(self, frame):
        self.original_shape = frame.shape[:2]
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)
    
    def postprocess(self, outputs, frame_shape):
        predictions = outputs[0].transpose(0, 2, 1)
        
        boxes = []
        confidences = []
        class_ids = []
        
        h, w = frame_shape[:2]
        scale_x = w / self.input_size
        scale_y = h / self.input_size
        
        for pred in predictions[0]:
            x_center, y_center, width, height, confidence = pred
            
            if confidence > self.conf_threshold:
                # SỬA: Dùng scale_x và scale_y đúng
                x_center = x_center * scale_x
                y_center = y_center * scale_y
                box_width = width * scale_x     # ✅ Đúng
                box_height = height * scale_y   # ✅ SỬA: Từ "height * h" thành "height * scale_y"
                
                # Convert center format sang corner format
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                
                # Clamp coordinates
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                box_width = min(int(box_width), w - x1)
                box_height = min(int(box_height), h - y1)
                
                boxes.append([x1, y1, box_width, box_height])
                confidences.append(float(confidence))
                class_ids.append(0)
        
        # NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes, 
                confidences, 
                self.conf_threshold, 
                self.iou_threshold
            )
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]
                confidences = [confidences[i] for i in indices]
                class_ids = [class_ids[i] for i in indices]
                
                # Chỉ giữ detection có confidence cao nhất nếu > 1
                if len(boxes) > 1:
                    max_conf_idx = np.argmax(confidences)
                    boxes = [boxes[max_conf_idx]]
                    confidences = [confidences[max_conf_idx]]
                    class_ids = [class_ids[max_conf_idx]]
        
        return boxes, confidences, class_ids
    
    def draw_detections(self, frame, boxes, confidences, class_ids):
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x, y, w, h = box
            label = f"{self.class_names[class_id]}: {conf:.2f}"
            
            # Vẽ bounding box màu xanh lá
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Vẽ label background
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y-text_h-baseline-10), (x+text_w+10, y), (0, 255, 0), -1)
            
            # Vẽ text
            cv2.putText(frame, label, (x+5, y-baseline-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Vẽ center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        
        return frame

# Khởi tạo
detector = YOLODetector(
    "best.onnx", 
    conf_threshold=0.55,
    iou_threshold=0.6
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit")

import time
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Inference
    input_tensor = detector.preprocess(frame)
    outputs = detector.session.run(None, {detector.input_name: input_tensor})
    boxes, confidences, class_ids = detector.postprocess(outputs, frame.shape)
    frame = detector.draw_detections(frame, boxes, confidences, class_ids)
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Display info
    info = f"FPS: {fps:.1f} | Detections: {len(boxes)}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('YOLO11 Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped")
