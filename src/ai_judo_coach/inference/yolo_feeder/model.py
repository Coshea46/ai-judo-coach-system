from ultralytics import YOLO

def load_yolo_model(yolo_model_path: str) -> YOLO:
    """
    Insantiates the yolo model using its .pt weights stored 
    at a given file path.
    """

    model = YOLO(yolo_model_path)

    return model