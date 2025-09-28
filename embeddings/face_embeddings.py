from deepface import DeepFace

def get_embedding(image_path, model_name="ArcFace"):
    """
    Extract face embedding using DeepFace backend
    """
    # aligned_face = functions.preprocess_face(img_path="face1.jpg", target_size=(112,112))
    rep = DeepFace.represent(img_path=image_path, model_name=model_name)[0]["embedding"]
    return rep
