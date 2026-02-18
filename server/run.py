"""
开发启动入口：python run.py
生产部署请使用：uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 4
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
