# Project Gemini: AI 스터디 집중력 분석기

이 문서는 Gemini와의 대화 내용을 기반으로, AI 스터디 집중력 분석기 프로젝트의 현재 상태를 요약합니다.

## 1. 프로젝트 목표

-   **서비스명**: AI 스터디 집중력 분석기
-   **핵심 기능**: 웹캠을 통해 사용자의 행동을 실시간으로 분석하여 학습 집중도를 측정하고 피드백을 제공한다.
-   **차별점**: '하품', '조는 행위' 등 **연속적인 동작(Action)**을 시계열 분석을 통해 파악하여 분석의 정확도를 높인다.

## 2. 아키텍처 개요

-   **처리 방식**: **프레임별 시계열 처리**
-   **통신 방식**: **WebSocket**을 이용한 실시간 스트리밍
-   **데이터 흐름**:
    1.  **프론트엔드**: **초당 10~20 프레임**을 백엔드로 실시간 스트리밍한다. (Base64 인코딩된 JPEG 이미지)
    2.  **백엔드 (1단계: 특징 추출)**: 각 프레임에서 MediaPipe를 사용해 얼굴/손의 **특징점(좌표)을 추출**한다.
    3.  **백엔드 (2단계: 단일 프레임 분석)**: 추출된 특징점을 기반으로 눈 상태(EAR), 머리 자세(HPE) 등 단일 프레임 기반의 행동 분석을 수행한다.
    4.  **백엔드 (3단계: 시계열 분석)**: 짧은 시간(예: 30프레임) 동안 쌓인 특징점들의 **연속적인 데이터(시계열)**를 PyTorch 모델(LSTM)에 입력하여 '하품', '눈 비빔' 등의 최종 동작을 판단한다.
    5.  **결과 반환**: 최종 분석 결과를 **JSON** 형식으로 프론트엔드로 전송한다.
-   **다중 사용자 지원**: 각 웹소켓 연결(사용자 세션)마다 독립적인 `AnalysisService` 인스턴스를 할당하여, 여러 사용자가 동시에 접속해도 서로의 분석에 영향을 주지 않도록 격리된 환경을 제공한다. 이는 FastAPI의 종속성 주입(Dependency Injection)과 팩토리 함수(Factory Function)를 활용하여 구현된다.

## 3. 기술 스택 및 API 명세

-   **백엔드 프레임워크**: **FastAPI**
-   **AI/CV 라이브러리**: MediaPipe (특징 추출), PyTorch (시계열 분석), scikit-learn (단일 프레임 분류)
-   **권장 Python 버전**: 3.10

### API 명세 (프론트엔드와의 약속)

-   **WebSocket 엔드포인트**: `wss://focushabit.site/ws/analysis` (또는 개발 환경에 따라 `ws://localhost:8000/ws/analysis`)
-   **프론트엔드 → 백엔드 데이터**: Base64 인코딩된 이미지 문자열 (JPEG 형식, 초당 10~20회)
-   **백엔드 → 프론트엔드 데이터**: 분석 결과 JSON 객체 (예시: 아래 '행동 분석 모듈 명세' 참조)

## 4. 백엔드 디렉터리 구조

```
Focus_habit_BE (루트 경로)/
│
├── .env
├── .gitignore
├── main.py                 # FastAPI 애플리케이션 진입점 (전역 AnalysisService 인스턴스 제거)
├── requirements.txt
│
├── api/
│   └── websockets.py       # 웹소켓 통신 핸들링, 연결별 AnalysisService 인스턴스 주입 및 호출
│
├── models/
│   ├── lstm_action_classifier.py # PyTorch LSTM 모델 구조 정의
│   └── checkpoints/              # 학습된 모델 가중치 및 MediaPipe 모델 파일
│       ├── best_lstm_model.pth
│       ├── face_landmarker.task
│       └── hand_landmarker.task
│
├── dependencies/
│   └── factories.py        # AnalysisService 인스턴스 생성을 위한 팩토리 함수 정의
│
└── services/
    ├── analysis_service.py     # 핵심 분석 로직 (MediaPipe, 단일 프레임 분석, 시계열 분석 통합), 지연 초기화 및 리소스 정리 로직 포함
    ├── data_service.py         # (현재 미사용 또는 확장 예정)
    ├── media_pipe_runner.py    # MediaPipe Face Landmarker 래퍼 클래스 (리소스 정리 close() 메서드 포함)
    ├── supabase_service.py     # (Supabase 연동 예정)
    └── modules/
        ├── eye_analyzer.py     # 눈 상태(EAR) 분석 모듈
        └── head_pose_analyzer.py # 머리 자세(HPE) 분석 모듈
```

## 5. 주요 기술 결정사항 및 개발 계획

### Q&A

-   **Q: 연속적인 동작(하품 등) 분석을 위한 데이터 처리 방식은?**
    -   **A: '프레임별 시계열 처리' 방식을 채택.** 프레임 단위로 특징(좌표)을 먼저 추출하고, 이 특징들의 시간적 순서를 신경망으로 분석함. 이 방식은 PyTorch 모델 확장성 및 프론트엔드와의 협업 편의성 면에서 '동영상 단위 처리' 방식보다 우수하다고 판단됨.

-   **Q: Base64/JPEG 변환 시 데이터 손실로 AI 인식률이 저하되지 않는가?**
    -   **A: 문제 없음.** JPEG 압축은 MediaPipe가 인식에 사용하는 **핵심적인 형태(shape), 외곽선(contour) 정보는 대부분 보존**하므로, 이는 데이터 전송량을 줄이는 표준적인 방식임.

## 6. 행동 분석 모듈 명세 (백엔드 → 프론트엔드 최종 출력 예시)

각 행동 분석 모듈은 MediaPipe로부터 추출된 랜드마크를 입력으로 받아 특정 행동을 분석하고 결과를 반환합니다. `AnalysisService`는 이 모듈들의 결과를 통합하여 최종 JSON 응답을 생성합니다.

### Module 1: Eye State Detection (EAR)
눈의 개폐 상태를 실시간으로 탐지합니다.

-   **NAME**: `eye_state_ear`
-   **DESCRIPTION**: 얼굴 랜드마크를 기반으로 눈 종횡비(Eye Aspect Ratio)를 계산하여 눈을 감았는지 떴는지 판별합니다.
-   **METHODOLOGY**:
    1.  MediaPipe Face Landmarker로부터 양쪽 눈 주변의 랜드마크 좌표를 수신합니다.
    2.  EAR 공식을 이용해 각 프레임의 눈 종횡비(EAR) 값을 계산합니다.
    3.  계산된 EAR 값을 `ear_threshold` 파라미터와 비교합니다.
    4.  EAR 값이 임계값 미만이면 'CLOSED', 이상이면 'OPEN' 상태로 판별합니다.
-   **INPUT**: `FaceLandmarks` (MediaPipe의 얼굴 랜드마크 배열)
-   **OUTPUT**: JSON (예: `{"status": "CLOSED", "ear_value": 0.15}`)
-   **CONFIGURABLE_PARAMETERS**: `--ear_threshold` (float): 눈을 감았다고 판단하는 EAR 값의 임계값. 기본값은 0.2.

### Module 2: Head Pose Estimation (HPE)
얼굴의 3차원 방향과 기울기를 추정합니다.

-   **NAME**: `head_pose_hpe`
-   **DESCRIPTION**: 얼굴의 3D 랜드마크를 사용하여 머리의 위아래(Pitch), 좌우(Yaw), 기울임(Roll) 각도를 계산합니다.
-   **METHODOLOGY**:
    1.  MediaPipe Face Landmarker로부터 3D 얼굴 랜드마크 좌표와 변환 행렬을 수신합니다.
    2.  주요 랜드마크(코, 턱, 양 미간 등)를 기준으로 3차원 공간상의 얼굴 방향 벡터를 정의합니다.
    3.  벡터 연산을 통해 Pitch, Yaw, Roll 회전 각도를 도(degree) 단위로 계산합니다.
-   **INPUT**: `FaceLandmarks` (MediaPipe의 3D 얼굴 랜드마크 배열)
-   **OUTPUT**: JSON (예: `{"pitch": 15.2, "yaw": -5.1, "roll": 2.5}`)
-   **CONFIGURABLE_PARAMETERS**: `--smoothing_factor` (float): 출력 각도의 노이즈를 줄이기 위한 평활화 계수. 0과 1 사이 값으로, 기본값은 0.5.

### 최종 출력 예시 (백엔드 → 프론트엔드)
```json
{
  "timestamp": 1752563960781,
  "eye_status": {
    "status": "OPEN",
    "ear_value": 0.4047
  },
  "head_pose": {
    "pitch": 14.77,
    "yaw": -62.37,
    "roll": 176.03
  },
  "prediction_result": {
    "timestamp": 1752563960781,
    "prediction": "focused",
    "confidence": 0.95
  }
}
```
Note: The `prediction_result` will only be present when the `SEQUENCE_LENGTH` (30 frames) is reached and a prediction is made. Otherwise, only `timestamp`, `eye_status`, and `head_pose` will be returned.

## 7. 다중 사용자 지원 및 리소스 관리

### 7.1 문제점: 워커 프로세스 간의 암시적 상태 공유

초기 구현에서는 `AnalysisService` 인스턴스가 Uvicorn의 다중 워커 프로세스 모델에서 예상치 못하게 공유되는 문제가 발생했습니다. 각 워커 프로세스는 시작 시 애플리케이션 코드를 로드하며, 이 과정에서 `AnalysisService` 인스턴스가 워커 수준에서 한 번 생성되고 해당 워커로 라우팅되는 모든 웹소켓 연결에 재사용되었습니다. 이로 인해 각 사용자의 세션이 독립적으로 격리되지 않고, 버퍼 및 MediaPipe 컨텍스트가 워커 내에서 공유되는 현상이 발생했습니다.

### 7.2 해결책: 팩토리 함수를 통한 명시적 인스턴스 관리

이 문제를 해결하기 위해 FastAPI의 종속성 주입(Dependency Injection) 시스템과 팩토리 함수(Factory Function)를 활용하여 `AnalysisService` 인스턴스를 명시적으로 연결별(Per-Connection)로 생성하도록 변경했습니다.

*   **팩토리 함수 (`get_analysis_service`) 도입:** `app/dependencies/factories.py`에 `get_analysis_service` 함수를 정의하여, 이 함수가 호출될 때마다 새로운 `AnalysisService` 인스턴스를 반환하도록 했습니다.
*   **`websocket_endpoint`에 팩토리 함수 주입:** `app/api/websockets.py`의 `websocket_endpoint`에서 `analysis_service: AnalysisService = Depends(get_analysis_service)`를 사용하여, 웹소켓 연결이 설정될 때마다 팩토리 함수가 호출되어 독립적인 `AnalysisService` 인스턴스가 주입되도록 보장했습니다.
*   **지연 초기화 및 명시적 리소스 정리:** `AnalysisService` 내에서 MediaPipe `MediaPipeRunner`를 지연 초기화하고, 웹소켓 연결 종료 시 `AnalysisService.close()` 및 `MediaPipeRunner.close()`를 통해 사용된 리소스를 명시적으로 해제하도록 구현하여 리소스 누수를 방지하고 진정한 격리를 달성했습니다.

### 7.3 교훈 및 주의사항

이 과정에서 얻은 주요 교훈은 다음과 같습니다.

*   **Uvicorn/Gunicorn 워커 모델의 이해:** 다중 워커 환경에서는 애플리케이션 코드가 워커 프로세스당 한 번 로드되므로, 상태를 가지는 서비스는 워커 수준의 전역 상태에 의존하지 않도록 설계해야 합니다.
*   **상태 저장 서비스의 종속성 주입 패턴:** 요청별/연결별 격리가 필요한 서비스의 경우, 팩토리 함수를 `Depends`와 함께 사용하는 것이 가장 신뢰할 수 있는 패턴입니다.
*   **외부 리소스 관리의 중요성:** MediaPipe와 같이 파이썬 외부의 C++ 라이브러리나 GPU 리소스를 사용하는 경우, 명시적인 `close()` 메서드 구현 및 호출을 통해 리소스 누수를 방지하고 시스템 안정성을 확보해야 합니다.
*   **디버깅의 중요성:** 복잡한 동시성 및 상태 관리 문제에서는 상세한 로깅과 문제 격리 전략(예: 코드 단순화)이 문제의 근본 원인을 파악하는 데 필수적입니다.
*   **배포 환경과 개발 환경의 차이:** 로컬 개발 환경에서 작동하는 코드가 다중 프로세스 프로덕션 환경에서 다르게 동작할 수 있음을 인지하고, 실제 배포 환경과 유사한 조건에서 테스트하는 것이 중요합니다.
