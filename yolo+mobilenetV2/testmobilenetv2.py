import cv2
import numpy as np
import tensorflow as tf
import time

# Load TFLite model
print("Loading MobileNetV2 TFLite model...")
interpreter = tf.lite.Interpreter(model_path="mobilenetv2_optimized.tflite")
interpreter.allocate_tensors()

# Lấy input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Input dtype: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"✓ Model loaded successfully\n")

# Class names: 0 = def_front, 1 = ok_front
class_names = ['def_front', 'ok_front']
colors = [(0, 0, 255), (0, 255, 0)]  # Đỏ cho def, Xanh cho ok

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting classification...")
print("Press 'q' to quit\n")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess toàn bộ frame
    img = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Get prediction
    pred_class = np.argmax(output[0])
    confidence = output[0][pred_class]
    class_name = class_names[pred_class]
    color = colors[pred_class]
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    # Vẽ thông tin lên frame
    # Background cho label
    label_text = f"{class_name}: {confidence:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    
    # Vẽ background rectangle lớn ở góc trên bên trái
    cv2.rectangle(frame, (10, 10), (text_w + 30, text_h + baseline + 30), color, -1)
    
    # Vẽ text trắng
    cv2.putText(frame, label_text, (20, text_h + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Vẽ border toàn màn hình theo màu class
    border_thickness = 15
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, border_thickness)
    
    # Hiển thị confidence scores cho cả 2 classes
    cv2.putText(frame, f"def_front: {output[0][0]:.3f}", (10, frame.shape[0] - 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"ok_front:  {output[0][1]:.3f}", (10, frame.shape[0] - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show frame
    cv2.imshow('MobileNetV2 Classification Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nClassification test stopped")
