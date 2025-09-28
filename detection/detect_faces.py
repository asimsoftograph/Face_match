import os
from ultralytics import YOLO
import cv2




def detect_and_crop(image_path, output_dir, model_path="E:\city_bank_face_match\models\yolov8n_100e_face.pt"):
    """
    Detect face using YOLOv8 and save cropped image
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    results = model(image_path)

    img = cv2.imread(image_path)
    crops = []
    for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img[y1:y2, x1:x2]
        crop_path = os.path.join(output_dir, f"crop_{i}.jpg")
        cv2.imwrite(crop_path, crop)
        crops.append(crop_path)
    return crops




# import os
# import cv2
# from ultralytics import YOLO

# def detect_and_crop(image_path, output_dir, model_path="E:\city_bank_face_match\models\yolo_v8n_finetuned_hand_signatures.pt"):
#     """
#     Detect signuture using YOLOv8 and save cropped signuture images.
#     Returns list of cropped file paths.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Load your trained YOLOv8 model
#     model = YOLO(model_path)

#     # Run detection
#     results = model(image_path)

#     # Load original image
#     img = cv2.imread(image_path)
#     crops = []

#     for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
#         x1, y1, x2, y2 = map(int, box[:4])
#         crop = img[y1:y2, x1:x2]

#         crop_path = os.path.join(output_dir, f"crop_{i+1}.jpg")
#         cv2.imwrite(crop_path, crop)
#         crops.append(crop_path)

#         print(f"[INFO] Saved crop {i+1}: {crop_path}")

#     return crops

# if __name__ == "__main__":
#     # 🔹 Example test run
#     test_image = "E:/city_bank_face_match/data/nid/nid_pic.png"   # <-- put one test image here
#     output_dir = "E:/city_bank_face_match/output/crops"

#     cropped_faces = detect_and_crop(test_image, output_dir)
#     print("Detected Crops:", cropped_faces)
#     print("✅ Cropped images saved in:", output_dir)