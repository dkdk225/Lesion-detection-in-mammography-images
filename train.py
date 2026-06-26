from ultralytics import YOLO
model = YOLO("yolov8s.pt")

model.train(
    data="/content/drive/MyDrive/Lesion detection/preprocessed_dataset3/dataset.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    # --- Augmentations ---
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.5,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0,

    #GOOGLE DRIVE SAVING:
    project="/content/drive/MyDrive/runs", # Creates a 'runs' folder in your Drive
    name="yolov8n_mammo"
)

metrics = model.val()
print("\n=== BENCHMARK ACCURACY RESULTS ===")
print(f"Class name:           {metrics.names[0]}")
print(f"mAP@50 (IoU 0.50):    {metrics.box.map50:.4f}")
print(f"mAP@50-95 (Strict):   {metrics.box.map:.4f}")
print(f"Precision (Accuracy): {metrics.box.mp:.4f}")
print(f"Recall (Sensitivity): {metrics.box.mr:.4f}")