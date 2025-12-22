from insightface.app import FaceAnalysis


def load_arcface_model():
    """
    Loads ArcFace (InsightFace) model for CPU inference.
    Includes face detection + alignment + embedding.
    """
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    # ctx_id = -1 → CPU
    app.prepare(ctx_id=-1, det_size=(640, 640))

    return app
