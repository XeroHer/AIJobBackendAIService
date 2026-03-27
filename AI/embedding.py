_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")  # faster + lighter
    return _model


def get_embedding(text: str):
    model = get_model()
    return model.encode(text, normalize_embeddings=True)