# Focus Habit: AI 기반 학습 집중도 분석기

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Focus Habit**은 웹캠을 통해 사용자의 학습 태도를 실시간으로 분석하여, 집중도 저하 행동을 감지하고 피드백을 제공하는 AI 기반 서비스입니다. 특히, 사용자별 행동 패턴에 적응하는 **개인화 모델**을 통해 분석의 정확도를 높인 것이 특징입니다.

## ✨ 주요 기능

-   **실시간 집중도 분석**: 웹캠 스트림을 통해 사용자의 눈 상태(EAR), 머리 자세(HPE) 등을 실시간으로 분석합니다.
-   **정확한 행동 인식 (CNN+GRU)**: 1D-CNN으로 특징을 추출하고 GRU로 시간적 맥락을 분석하는 하이브리드 모델을 사용하여 '하품', '조는 행위' 등 연속적인 동작을 정확하게 판단합니다.
-   **사용자 맞춤형 개인화 모델**: Supabase와 연동하여 사용자별로 미세 조정된(fine-tuned) 모델을 동적으로 로드합니다. 개인화 모델이 없는 경우, 강력한 기본(Baseline) 모델로 분석을 수행합니다.
-   **확장성 있는 아키텍처**: FastAPI의 종속성 주입(Dependency Injection)을 활용하여 WebSocket 연결마다 독립적인 분석 서비스를 제공하므로, 다중 사용자 환경에서도 안정적인 성능을 보장합니다.

## ⚙️ 모델 서빙 아키텍처 (Model Serving Architecture)

Focus Habit의 모델 서빙은 WebSocket 연결을 기반으로 실시간 추론(Inference)을 수행하는 방식으로 설계되었습니다. 전체 흐름은 다음과 같습니다.

1.  **연결 및 서비스 초기화**: 클라이언트가 `ws://.../ws/analysis` 엔드포인트로 WebSocket 연결을 요청하면, FastAPI는 해당 연결 전용 `AnalysisService` 인스턴스를 생성합니다. 이 때 `user_id`가 제공되면, 서비스 내부에 AI 추론을 담당하는 `PersonalizedModelRunner`가 함께 초기화됩니다.

2.  **모델 동적 로딩**: `PersonalizedModelRunner`는 초기화 과정에서 Supabase 스토리지에 해당 유저의 **개인화 모델**이 있는지 확인하고, 존재하면 모델 파일(`.pth`)과 정규화 스케일러(`.pkl`)를 다운로드하여 메모리에 로드합니다. 개인화 모델이 없으면 사전에 학습된 **기본(Baseline) 모델**을 로드합니다.

3.  **실시간 추론 (Real-time Inference)**:
    -   클라이언트는 웹캠 이미지를 프레임 단위로 서버에 전송합니다.
    -   `AnalysisService`는 각 프레임에서 MediaPipe를 이용해 얼굴 특징점을 추출하고, 1차 분석(눈, 머리 상태)을 수행합니다.
    -   분석된 데이터는 `PersonalizedModelRunner` 내부의 버퍼에 순차적으로 쌓입니다.
    -   버퍼에 데이터가 일정 크기(`window_size`, 예: 25프레임) 이상 쌓이면, `TimeSeriesCNNGRU` 모델이 버퍼의 데이터를 입력으로 받아 **추론**을 수행하고, 사용자의 현재 집중 상태에 대한 예측 확률을 계산합니다.

4.  **결과 반환**: 추론을 통해 생성된 최종 예측 결과(집중도 점수, 신뢰도 등)는 다른 분석 데이터와 함께 JSON 형식으로 클라이언트에 다시 전송됩니다. 이 과정이 실시간으로 반복되어 연속적인 피드백을 제공합니다.

## 🛠️ 기술 스택

-   **Backend**: FastAPI
-   **AI/CV**: PyTorch, MediaPipe, Scikit-learn
-   **Database/Storage**: Supabase (모델 및 데이터 저장)
-   **Language**: Python 3.10
-   **Deployment**: Docker, Nginx

## 🚢 배포 및 모델 서빙 (Deployment & Model Serving)

이 프로젝트는 Docker를 통해 컨테이너 환경에서 실행되도록 설계되었습니다. `docker-compose.yml` 파일은 프로덕션 환경에 필요한 모든 서비스를 정의하고 관리합니다.

-   **`app`**: FastAPI 백엔드 애플리케이션을 실행하는 메인 서비스입니다.
-   **`nginx`**: Nginx 리버스 프록시 서버입니다. 외부의 HTTP(80) 및 HTTPS(443) 요청을 받아 FastAPI `app` 서비스로 전달합니다. 또한, SSL/TLS 인증서를 처리하여 HTTPS 통신을 책임집니다.
-   **`certbot`**: Let's Encrypt를 사용하여 SSL 인증서를 자동으로 발급하고 갱신하는 서비스입니다.
-   **`dozzle`**: 실행 중인 모든 컨테이너의 로그를 웹 인터페이스를 통해 실시간으로 확인할 수 있는 경량 로그 뷰어입니다.

### Docker를 이용한 실행

```bash
# 프로젝트 루트 디렉터리에서 다음 명령어를 실행하여 모든 서비스를 빌드하고 실행합니다.
docker-compose up --build -d
```

## 📂 디렉터리 구조

```
Focus_habit_BE/
└── app/
    ├── api/
    │   ├── training_api.py         # 개인화 모델 학습 API
    │   └── websockets.py           # WebSocket 핸들링 및 분석 서비스 주입
    ├── dependencies/
    │   └── factories.py            # AnalysisService 인스턴스 생성 팩토리
    ├── models/
    │   ├── realtimemodel_gru.py    # TimeSeriesCNNGRU 모델 및 PersonalizedModelRunner 정의
    │   ├── lstm_action_classifier.py # (구 버전)
    │   └── checkpoints/              # 기본 모델, 스케일러, MediaPipe 모델 파일
    ├── services/
    │   ├── analysis_service.py     # 핵심 분석 로직 통합 서비스
    │   ├── media_pipe_runner.py    # MediaPipe 모델 래퍼
    │   ├── personal_train.py       # 개인화 모델 학습/저장 로직
    │   ├── supabase_service.py     # Supabase 연동 서비스
    │   └── modules/
    │       ├── eye_analyzer.py     # 눈 상태(EAR) 분석
    │       └── head_pose_analyzer.py # 머리 자세(HPE) 분석
    ├── Dockerfile
    ├── main.py                     # FastAPI 애플리케이션 진입점
    └── requirements.txt
```

## 🚀 시작하기

### 1. 사전 요구사항

-   Python 3.10 이상
-   pip

### 2. 설치

```bash
# 1. 프로젝트 클론
git clone https://github.com/your-username/Focus_habit_BE.git
cd Focus_habit_BE

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate # Linux/macOS
# venv\Scripts\activate # Windows

# 3. 의존성 설치
pip install -r app/requirements.txt
```

### 3. 실행

```bash
# app 디렉터리에서 uvicorn 서버 실행
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

서버가 실행되면 `ws://localhost:8000/ws/analysis` 엔드포인트를 통해 웹캠 스트리밍을 시작할 수 있습니다.

## 🔌 API 명세

-   **WebSocket Endpoint**: `ws://localhost:8000/ws/analysis`
-   **프론트엔드 → 백엔드**: Base64로 인코딩된 JPEG 이미지 문자열을 초당 10~20회 전송합니다.
-   **백엔드 → 프론트엔드**: 분석 결과를 담은 JSON 객체를 실시간으로 전송합니다.

    **예시 응답:**
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
        "prediction": 95.0, // 집중도를 0-100 사이 값으로 표현
        "confidence": 0.95
      }
    }
    ```
    > `prediction_result`는 추론이 완료되었을 때만 포함됩니다.