import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

@app.get("/token")
async def get_token():
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    url = os.getenv("LIVEKIT_URL")
    
    if not api_key or not api_secret or not url:
        return {"error": "Missing environment variables"}

    grant = api.VideoGrant(room_join=True, room="terminator-room")
    token = api.AccessToken(api_key, api_secret, grant=grant)
    token.identity = "human-user"
    token.name = "Human"
    
    jwt = token.to_jwt()
    return {"token": jwt, "url": url}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

