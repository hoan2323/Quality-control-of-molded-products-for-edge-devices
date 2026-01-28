from ultralytics import YOLO

model = YOLO("best.pt")

# Export ONNX với FP16 
model.export(
    format="onnx",
    half=True,      
    imgsz=320,
    simplify=True,   
    dynamic=False   
)
