FROM python:3.10-bullseye

# OpenCV 의존성 수동설치
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일들을 컨테이너에 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 앱 코드를 컨테이너에 복사
COPY . /app

CMD ["uvicorn", "main:app", "--port", "8000", "--host", "0.0.0.0", "--log-level", "debug"]