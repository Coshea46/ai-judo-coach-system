from fastapi import(
    FastAPI 
)
from fastapi.middleware.cors import CORSMiddleware

from ai_judo_coach.api.routes import router


app = FastAPI()

# allowed origins for CORS
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount the routing paths
app.include_router(router, tags=["Routes"])


