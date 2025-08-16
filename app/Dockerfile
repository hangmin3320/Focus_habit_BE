FROM python:3.10-bullseye

# OpenCV 의존성 수동설치
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 작업 디렉토리 설정
WORKDIR /usr/src/app

ENV PYTHONPATH=.

# 의존성 파일들을 컨테이너에 복사
COPY requirements.txt .

# PyTorch CPU 버전 설치 (별도 인덱스 사용)
# typing-extensions 선행 설치 (PyTorch 종속성 문제 해결)
RUN pip install typing-extensions==4.12.2

# PyTorch CPU 버전 설치 (별도 인덱스 사용)
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 앱 코드를 컨테이너에 복사
COPY . .

CMD ["uvicorn", "app.main:app", "--port", "8000", "--host", "0.0.0.0", "--log-level", "debug"]