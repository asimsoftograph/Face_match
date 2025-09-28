import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def calculate_accuracy(similarities):
    avg = np.mean(similarities)
    return round(avg * 100, 2)
