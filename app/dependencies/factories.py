from ..services.analysis_service import AnalysisService
from fastapi import Depends, WebSocket

# 이 함수는 user_id를 웹소켓 쿼리 파라미터로부터 직접 추출하는 역할을 합니다.
# 하지만 현재 구조에서는 FastAPI가 자동으로 매핑해주므로 직접 사용할 필요는 없습니다.
# 단지 명시적으로 어떤 타입의 의존성인지 보여주기 위해 남겨둘 수 있습니다.
def get_user_id_from_websocket(websocket: WebSocket) -> str:
    return websocket.query_params.get("user_id")

def get_analysis_service() -> AnalysisService:
    """
    새로운 AnalysisService 인스턴스를 생성하는 팩토리 함수입니다.
    각 의존성 주입 시마다 새로운 인스턴스를 보장합니다.
    
    향후 이 함수는 user_id를 인자로 받아 AnalysisService(user_id=user_id)를 생성하게 됩니다.
    """
    print("Factory: Creating new AnalysisService instance.")
    # TODO: AnalysisService.__init__에 user_id가 추가되면 아래 코드를 활성화해야 합니다.
    # return AnalysisService(user_id=user_id)
    return AnalysisService()