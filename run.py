import uvicorn
from server.api.endpoints.routes import add_routes
from settings import Config
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
app = add_routes()

print(Config.host, Config.port)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=Config.host,
        port=Config.port,
        workers=1,
    )
