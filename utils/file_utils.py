import os

def list_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
