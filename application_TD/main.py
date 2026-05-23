from fastapi import FastAPI
import uvicorn #a high‑performance web server for Python, Think of it as the “engine” that runs your FastAPI app 

# from admin import admin_app
from model import model_app
# from dashboard import dashboard_app
from utils import load_config

config = load_config()
app = FastAPI()
app.mount("/model", model_app)
# app.mount("/admin", admin_app)
# app.mount("/dashboard", dashboard_app)

#health check endpoint, Kubernetes liveness probes, monitoring systems, API readiness checks
@app.get("/healthz/")
async def healthz(): #it shows that this a async function, it allows the server to handle many resquests efficiently 
    return {"status": "ok"}

if __name__ == "__main__":

    worker_count = config.get("worker_count", 1) #number of parallel worker threads
    if worker_count == 1:
        uvicorn.run(app, host="0.0.0.0", port=config.get("server_port", 8765), log_config="log.ini")
    else:
        import sys
        import subprocess
        
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", str(config.get("server_port", 8765)),
            "--workers", str(worker_count),
            "--log-config", "log.ini"
        ])