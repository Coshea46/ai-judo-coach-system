
class InvalidVideoError(Exception):
    """Raised when a video file cannot be read or parsed"""
    pass



class InvalidFrameIndicesError(Exception):
    """
    Raised when the indices of the frames desired
    in a video are outwith the bounds of the video
    """
    pass