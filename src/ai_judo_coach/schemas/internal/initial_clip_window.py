from pydantic import BaseModel

class InitialClipWindow(BaseModel):
    
    start_time: float       # window start, seconds
    end_time: float         # window end, seconds
    window_id: int              # unique id for window object