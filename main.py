import os
import json
from config.settings import FORM_PDF_PATH, NID_IMAGE_PATH, CROPS_PATH, SIMILARITY_THRESHOLD, PDF_TEMP_IMAGES_PATH
from detection.detect_faces import detect_and_crop
from embeddings.face_embeddings import get_embedding
from similarity.similarity_check import cosine_similarity, calculate_accuracy
from utils.file_utils import list_images, convert_pdf_to_images

def process_input_path(input_path, output_folder):
    """
    Processes an input path, converting PDFs to images if necessary,
    and returns a list of image file paths.
    """
    if input_path.lower().endswith(".pdf"):
        print(f"Converting PDF {input_path} to images...")
        return convert_pdf_to_images(input_path, output_folder)
    elif os.path.isdir(input_path):
        print(f"Listing images in folder {input_path}...")
        return list_images(input_path)
    elif os.path.isfile(input_path) and input_path.lower().endswith((".jpg", ".png", ".jpeg")):
        print(f"Processing single image file {input_path}...")
        return [input_path]
    else:
        raise ValueError(f"Unsupported input path: {input_path}. Must be a folder, image file, or PDF.")

def main():
    results = {}

    # Step 1: Prepare form images (from PDF)
    form_image_paths = process_input_path(FORM_PDF_PATH, os.path.join(PDF_TEMP_IMAGES_PATH, "forms"))
    nid_image_paths = process_input_path(NID_IMAGE_PATH, os.path.join(PDF_TEMP_IMAGES_PATH, "nid"))

    print("Detecting faces in forms...")
    form_crops = []
    for i, f_path in enumerate(form_image_paths):
        crops = detect_and_crop(f_path, os.path.join(CROPS_PATH, f"form{i+1}"))
        if crops:
            form_crops.append(crops[0])  # assume one face per form
        else:
            print(f"No face detected in {f_path}, skipping.")

    print("Detecting face in NID...")
    nid_face = None
    if nid_image_paths:
        nid_crops = detect_and_crop(nid_image_paths[0], os.path.join(CROPS_PATH, "nid"))
        if nid_crops:
            nid_face = nid_crops[0]
        else:
            print(f"No face detected in NID image {nid_image_paths[0]}. Cannot proceed with similarity check.")
            return # Exit if NID face not found
    else:
        print("No NID image provided. Cannot proceed with similarity check.")
        return # Exit if NID image not provided

    # Step 2: Extract embeddings
    nid_embedding = get_embedding(nid_face)
    similarities = []

    for i, face in enumerate(form_crops):
        form_embedding = get_embedding(face)
        sim = cosine_similarity(form_embedding, nid_embedding)
        similarities.append(sim)
       
        results[f"form{i+1}"] = {
                       "similarity": round(float(sim), 2),
                       "match": bool(sim > SIMILARITY_THRESHOLD)
          }

        print(f"[INFO] Form {i+1} similarity: {sim:.4f} - Match: {sim > SIMILARITY_THRESHOLD}")
    # Step 3: Calculate final accuracy
    if similarities:
        final_accuracy = calculate_accuracy(similarities)
        results["final_accuracy"] = float(final_accuracy)
        results["all_match"] = bool(all(sim > SIMILARITY_THRESHOLD for sim in similarities))
    else:
        results["final_accuracy"] = 0.0
        results["all_match"] = False
        print("No similarities to calculate accuracy for.")
 

    # Step 4: Save results
    os.makedirs("output", exist_ok=True)
    with open("output/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Results saved in output/results.json")

if __name__ == "__main__":
    main()
