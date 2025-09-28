import os
from ultralytics import YOLO
import cv2
from pdf2image import convert_from_path
import logging
model_path="models/yolov8n_100e_face.pt"

# Optional: Set up logging for debugging
logging.basicConfig(level=logging.INFO)

def pdf_to_images(pdf_path, output_dir="output/pdf_images/"):
    """
    Convert a PDF file into a list of image file paths (one per page).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    try:
        # Convert PDF pages to PIL images
        images = convert_from_path(pdf_path, dpi=300, thread_count=1)  # thread_count avoids poppler issues
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images: {e}")

    image_paths = []
    for i, img in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i+1}.jpg")
        img.save(image_path, "JPEG", quality=95)
        image_paths.append(image_path)
        logging.info(f"Saved page {i+1} as {image_path}")
    
    return image_paths


def detect_and_crop(image_path, output_dir,model=model_path):
    """
    Detect faces in an image using a YOLOv8 model and save cropped face regions.
    Returns list of cropped image paths.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        model = YOLO(model_path)
        results = model(image_path)
    except Exception as e:
        raise RuntimeError(f"YOLO inference failed: {e}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image with OpenCV: {image_path}")

    crops = []
    boxes = results[0].boxes
    if len(boxes) == 0:
        logging.warning(f"No faces detected in {image_path}")
        return crops  # empty list

    for i, box in enumerate(boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box[:4])
        # Ensure coordinates are within image bounds
        h, w = img.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            logging.warning(f"Invalid crop coordinates in {image_path}: ({x1},{y1}) -> ({x2},{y2})")
            continue

        crop = img[y1:y2, x1:x2]
        crop_path = os.path.join(output_dir, f"crop_{i}.jpg")
        success = cv2.imwrite(crop_path, crop)
        if success:
            crops.append(crop_path)
            logging.info(f"Saved crop {i} to {crop_path}")
        else:
            logging.error(f"Failed to save crop {i} for {image_path}")

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