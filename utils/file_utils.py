import os
import fitz  # PyMuPDF

def list_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

def convert_pdf_to_images(pdf_path, output_folder):
    """
    Converts each page of a PDF file into a JPG image.
    Returns a list of paths to the generated images.
    """
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img_path = os.path.join(output_folder, f"page_{i+1}.jpg")
        pix.save(img_path)
        image_paths.append(img_path)
    doc.close()
    return image_paths
