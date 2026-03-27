_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading model...")
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Model loaded ✅")
        except Exception as e:
            print("Model load failed ❌:", e)
            return None
    return _model


def get_embedding(text: str):
    model = get_model()
    if model is None:
        return [0.0] * 384  # fallback so app never crashes
    return model.encode(text, normalize_embeddings=True)