from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from settings import Config

origins = [
    "http://localhost",
    "http://localhost:80",
]

app = FastAPI(debug=Config.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

