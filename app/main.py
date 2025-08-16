from fastapi import FastAPI
from app.api.websockets import router as websocket_router
from app.api.training_api import router as training_router # training_api 라우터 import
from app.services.analysis_service import AnalysisService # Changed from FocusAnalysisService

app = FastAPI(
    title="AI Study Focus Analyzer Backend",
    description="Real-time analysis of user focus using webcam feed.",
    version="0.1.0",
)

# 웹소켓 라우터 포함
app.include_router(websocket_router)
# 학습 API 라우터 포함
app.include_router(training_router, prefix="/api/v1", tags=["Training"])


@app.get("/", tags=["Health Check"])
async def read_root():
    return {"message": "AI Study Focus Analyzer Backend is running"}


# 애플리케이션 시작/종료 이벤트 핸들러
@app.on_event("startup")
async def startup_event():
    print("Application startup...")
    # AnalysisService is now instantiated per WebSocket connection, no global instance needed.


@app.on_event("shutdown")
async def shutdown_event():
    print("Application shutdown...")
    # Per-connection AnalysisService instances handle their own resource cleanup.

