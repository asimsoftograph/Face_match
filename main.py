import os
import json
from config.settings import FORM_IMAGES_PATH, NID_IMAGE_PATH, CROPS_PATH, SIMILARITY_THRESHOLD
from detection.detect_faces import detect_and_crop
from embeddings.face_embeddings import get_embedding
from similarity.similarity_check import cosine_similarity, calculate_accuracy
from utils.file_utils import list_images

def main():
    results = {}

    # Step 1: Detect and crop faces from forms
    form_files = list_images(FORM_IMAGES_PATH)
    nid_file = NID_IMAGE_PATH

    print("Detecting faces in forms...")
    form_crops = []
    for i, f in enumerate(form_files):
        crops = detect_and_crop(f, os.path.join(CROPS_PATH, f"form{i+1}"))
        form_crops.append(crops[0])  # assume one face per form

    print("Detecting face in NID...")
    nid_crops = detect_and_crop(nid_file, os.path.join(CROPS_PATH, "nid"))
    nid_face = nid_crops[0]

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
    final_accuracy = calculate_accuracy(similarities)
    results["final_accuracy"] = float(final_accuracy)
    results["all_match"] = bool(all(sim > SIMILARITY_THRESHOLD for sim in similarities))

 

    # Step 4: Save results
    os.makedirs("output", exist_ok=True)
    with open("output/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Results saved in output/results.json")

if __name__ == "__main__":
    main()








