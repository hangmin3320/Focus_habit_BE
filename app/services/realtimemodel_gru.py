import os, json, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, threading

CKPT_DIR = "app/models/checkpoint"
TEMPERATURE = 1.0

class TimeSeriesCNNGRU(nn.Module):
    """
    1D-CNN + GRU 기반의 이진 분류 모델.

    Args:
        input_channels (int): 입력 채널 수(피처 개수).
        window_size (int): 시퀀스 길이.
        dropout_rate (float): 드롭아웃 확률.
        k1 (int): 첫 번째 컨볼루션 커널 크기.
        k2 (int): GRU hidden size.
        two_convs (bool): 두 번째 컨볼루션 블록 사용 여부.

    Input:
        x (torch.Tensor): (B, T, C) 형태.

    Output:
        torch.Tensor: (B, 1) 로짓.
    """
    def __init__(self, input_channels, window_size=25, dropout_rate=0.3, k1=3, k2=32, two_convs=True):
        super().__init__()
        c=32; ks=k1; pad=ks//2
        self.two_convs=two_convs
        self.conv1=nn.Conv1d(input_channels,c,kernel_size=ks,padding=pad,bias=False)
        self.bn1=nn.BatchNorm1d(c)
        self.relu=nn.ReLU()
        self.pool1=nn.MaxPool1d(2)
        if self.two_convs:
            self.conv2=nn.Conv1d(c,c,kernel_size=ks,padding=pad,bias=False)
            self.bn2=nn.BatchNorm1d(c)
            self.pool2=nn.MaxPool1d(2)
        self.gru=nn.GRU(input_size=c,hidden_size=k2,num_layers=1,batch_first=True,bidirectional=False)
        self.dropout=nn.Dropout(dropout_rate)
        self.fc=nn.Linear(k2,1)

    def forward(self,x):
        """
        순전파.

        Args:
            x (torch.Tensor): (B, T, C)

        Returns:
            torch.Tensor: (B, 1) 로짓.
        """
        x=x.permute(0,2,1)
        x=self.pool1(self.relu(self.bn1(self.conv1(x))))
        if self.two_convs:
            x=self.pool2(self.relu(self.bn2(self.conv2(x))))
        x=x.permute(0,2,1)
        _,h=self.gru(x)
        h=self.dropout(h[-1])
        return self.fc(h)


def ensure_cols(df, cols):
    """
    지정 컬럼이 없으면 0.0 값으로 생성.

    Args:
        df (pd.DataFrame): 입력 데이터프레임.
        cols (list[str]): 보장할 컬럼 목록.

    Returns:
        pd.DataFrame: 컬럼이 보장된 데이터프레임.
    """
    for c in cols:
        if c not in df.columns: df[c]=0.0
    return df


def feature_engineer(df, base=('ear','pitch','yaw','roll'), username="USER"):
    """
    시계열 파생 피처 생성 및 정렬/결측 처리.

    생성 피처:
        - 각 base의 1차 차분(_diff)
        - 이동평균5(_mean_5), 이동표준편차5(_std_5)
        - blink_count(최근5 프레임)
        - angle_magnitude(pitch/yaw/roll의 변화량 벡터 크기)

    Args:
        df (pd.DataFrame): 원본 DF.
        base (tuple[str]): 기본 피처명.
        username (str): prefix 기본값.

    Returns:
        Tuple[pd.DataFrame, list[str]]: (가공 DF, 피처 리스트)
    """
    df=ensure_cols(df,list(base)+['eye_status','prefix'])
    if 'timestamp_ms' not in df.columns: df['timestamp_ms']=np.arange(len(df))
    if 'prefix' not in df.columns: df['prefix']=username
    df=df.sort_values(['prefix','timestamp_ms'])
    for f in base:
        if f not in df.columns: df[f]=0.0
    for f in base:
        df[f'{f}_diff']=df.groupby('prefix')[f].diff().fillna(0)
        m=df.groupby('prefix')[f].rolling(window=5,min_periods=1).mean().reset_index(level=0,drop=True)
        s=df.groupby('prefix')[f].rolling(window=5,min_periods=1).std().fillna(0).reset_index(level=0,drop=True)
        df[f'{f}_mean_5']=m
        df[f'{f}_std_5']=s
    df['eye_status_numeric']=df['eye_status'].map({'OPEN':1,'CLOSED':0}).fillna(0)
    t=df.groupby('prefix')['eye_status_numeric'].diff().eq(-1)
    df['blink_count']=t.rolling(window=5,min_periods=1).sum().fillna(0).reset_index(level=0,drop=True)
    df['angle_magnitude']=np.sqrt(df['pitch_diff']**2+df['yaw_diff']**2+df['roll_diff']**2)
    feats=list(base)+[f'{f}_diff' for f in base]+[f'{f}_mean_5' for f in base]+[f'{f}_std_5' for f in base]+['blink_count','angle_magnitude']
    df[feats]=df[feats].replace([np.inf,-np.inf],0).fillna(0)
    return df,feats


def compute_stride(window_size, overlap=0.5):
    """
    겹침 비율에 따른 stride 계산.

    Args:
        window_size (int): 윈도우 길이.
        overlap (float): 0~1, 예: 0.5면 50% 겹침.

    Returns:
        int: stride(최소 1).
    """
    return max(1,int(window_size*(1-overlap)))


def majority_smooth(labels, k=3):
    """
    과반 스무딩(선형 스트림에서 최근 k개 평균 반올림).

    Args:
        labels (Iterable[int]): 0/1 시퀀스.
        k (int): 윈도우 길이. k<=1이면 원본 반환.

    Returns:
        list[int]: 스무딩된 시퀀스.
    """
    if k is None or k<=1: return labels
    from collections import deque
    buf=deque(maxlen=k); out=[]
    for v in labels:
        buf.append(v)
        out.append(int(round(np.mean(buf))))
    return out


class PersonalizedModelRunner:
    """
    개인화 설정을 반영해 모델을 로드하고, 스트림 JSON을 누적/추론하는 런타임.

    동작:
        - JSON payload를 내부 버퍼에 누적
        - 전처리/스케일링 후 슬라이딩 윈도우 단위로 추론
        - 옵션에 따라 퍼센트/컨피던스로 구성된 JSON 레코드 반환

    Args:
        username (str): 사용자 ID. 개인화 자원 파일명 접두에 사용.
        use_personal (bool): 개인화 모델/스케일러가 있으면 우선 사용.

    Files:
        {CKPT_DIR}/{username}_model.pth
        {CKPT_DIR}/{username}_scaler.pkl
        {CKPT_DIR}/{username}_header.json
        기본값 미존재 시 {CKPT_DIR}/model.pth, scaler.pkl 사용
    """
    def __init__(self, username, use_personal=True):
        self.username=username
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.header_path=os.path.join(CKPT_DIR,f"{username}_header.json")
        self.model_path=os.path.join(CKPT_DIR,f"{username}_model.pth") if use_personal and os.path.isfile(os.path.join(CKPT_DIR,f"{username}_model.pth")) else os.path.join(CKPT_DIR,"model.pth")
        self.scaler_path=os.path.join(CKPT_DIR,f"{username}_scaler.pkl") if use_personal and os.path.isfile(os.path.join(CKPT_DIR,f"{username}_scaler.pkl")) else os.path.join(CKPT_DIR,"scaler.pkl")

        self.window_size=25
        self.overlap=0.5
        self.header_feats=None
        self.threshold=0.5
        self.smooth_k=None
        self.temperature=TEMPERATURE
        if os.path.isfile(self.header_path):
            with open(self.header_path,'r',encoding='utf-8') as f:
                hdr=json.load(f)
            self.window_size=int(hdr.get("window_size",25))
            self.overlap=float(hdr.get("overlap",0.5))
            self.header_feats=hdr.get("features",None)
            self.threshold=float(hdr.get("threshold",0.5))
            self.smooth_k=hdr.get("smooth_k",None)
            self.temperature=float(hdr.get("temperature",TEMPERATURE))

        with open(self.scaler_path,'rb') as f: self.scaler=pickle.load(f)

        self.features=None
        self.model=None
        self.lock=threading.Lock()

        self.buffer = pd.DataFrame({
            'timestamp_ms': pd.Series(dtype='float64'),
            'eye_status':   pd.Series(dtype='object'),
            'ear':          pd.Series(dtype='float64'),
            'pitch':        pd.Series(dtype='float64'),
            'yaw':          pd.Series(dtype='float64'),
            'roll':         pd.Series(dtype='float64'),
            'prefix':       pd.Series(dtype='object'),
        })

        self._load_model()

    def _load_model(self):
        """
        피처 목록 확정 후 모델 구조 생성 및 가중치 로드, eval 모드 전환.
        """
        if self.header_feats is None:
            tmp=pd.DataFrame([{"timestamp_ms":0,"eye_status":"OPEN","ear":0.0,"pitch":0.0,"yaw":0.0,"roll":0.0,"prefix":self.username}])
            tmp,feats=feature_engineer(tmp,username=self.username)
            self.features=feats
        else:
            self.features=self.header_feats
        self.model=TimeSeriesCNNGRU(len(self.features),self.window_size,0.3,3,32,True).to(self.device)
        sd=torch.load(self.model_path,map_location=self.device)
        self.model.load_state_dict(sd)
        self.model.eval()

    def _append_json(self, payload):
        """
        입력 JSON payload를 표준 스키마로 변환해 내부 버퍼에 누적.

        Args:
            payload (str|bytes|list|dict): 파일 경로, JSON 문자열, 또는 객체.
        """
        if isinstance(payload,str) and os.path.isfile(payload):
            with open(payload,'r',encoding='utf-8') as f: data=json.load(f)
        elif isinstance(payload,(str,bytes)):
            data=json.loads(payload)
        else:
            data=payload
        rows=[]
        for it in data:
            ts=it.get('timestamp',it.get('timestamp_ms',None))
            es=it.get('eye_status',{})
            hp=it.get('head_pose',{})
            ear_val=es.get('ear_value',es.get('ear',0.0))
            rows.append({
                'timestamp_ms': ts if ts is not None else np.nan,
                'eye_status': es.get('status','OPEN'),
                'ear': float(ear_val) if ear_val is not None else 0.0,
                'pitch': float(hp.get('pitch',0.0)),
                'yaw': float(hp.get('yaw',0.0)),
                'roll': float(hp.get('roll',0.0)),
                'prefix': self.username
            })
        df=pd.DataFrame(rows)
        if 'timestamp_ms' not in df.columns:
            df['timestamp_ms']=np.arange(len(df))
        if self.buffer.empty:
            self.buffer = df.copy()
        else:
            self.buffer = pd.concat([self.buffer, df], ignore_index=True)

    def push_and_infer(self, json_payload, threshold=None, return_prob=False, smooth_k=None, return_json=False):
        """
        JSON payload를 누적하고 슬라이딩 윈도우 단위로 추론.

        스케일링/윈도우 처리 후 각 윈도우 마지막 타임스탬프 기준으로 결과 1건 생성.
        return_json=True이면 아래 스키마로 반환:
            {
              "timestamp": <int>,
              "eye_status": {"status": <str>, "ear_value": <float>},
              "head_pose": {"pitch": <float>, "yaw": <float>, "roll": <float>},
              "prediction_result": {
                "timestamp": <int>,
                "prediction": <float:0~100>,  # 집중도 퍼센트
                "confidence": <float:0~1>     # max(p, 1-p)
              }
            }

        Args:
            json_payload (str|bytes|list|dict): 입력 JSON.
            threshold (float|None): 이진 예측 임계값. None이면 헤더의 threshold.
            return_prob (bool): True면 (preds, probs) 반환.
            smooth_k (int|None): 다수결 스무딩 길이. None이면 헤더의 smooth_k.
            return_json (bool): True면 JSON 레코드 리스트 반환.

        Returns:
            list|tuple: return_json=True → list[dict],
                        return_prob=True → (preds, probs),
                        그 외 → preds
        """
        with self.lock:
            self._append_json(json_payload)
            df=self.buffer.copy()
            df,feats=feature_engineer(df,username=self.username)
            if self.header_feats is not None:
                for c in self.header_feats:
                    if c not in df.columns: df[c]=0.0
                feats=self.header_feats
            df[feats]=self.scaler.transform(df[feats])

            stride=compute_stride(self.window_size,self.overlap)
            preds=[]; probs=[]; records=[]

            thr = threshold if threshold is not None else self.threshold
            use_smooth_k = smooth_k if smooth_k is not None else self.smooth_k

            for p in df['prefix'].unique():
                g=df[df['prefix']==p].sort_values('timestamp_ms')
                if len(g)<self.window_size: continue
                arr=g[feats].values
                for i in range(0,len(g)-self.window_size+1,stride):
                    x=arr[i:i+self.window_size][None,...]
                    x=torch.tensor(x,dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        logit=self.model(x).squeeze(-1)
                        pr=torch.sigmoid(logit/self.temperature).item()
                    probs.append(pr)
                    pred=1 if pr>=thr else 0
                    preds.append(pred)

                    if return_json:
                        ts=int(g['timestamp_ms'].iloc[i+self.window_size-1])
                        eye=str(g['eye_status'].iloc[i+self.window_size-1])
                        ear=float(g['ear'].iloc[i+self.window_size-1])
                        pitch=float(g['pitch'].iloc[i+self.window_size-1])
                        yaw=float(g['yaw'].iloc[i+self.window_size-1])
                        roll=float(g['roll'].iloc[i+self.window_size-1])
                        conf=float(max(pr,1.0-pr))
                        rec={
                            "timestamp": ts,
                            "eye_status": {"status": eye, "ear_value": ear},
                            "head_pose": {"pitch": pitch, "yaw": yaw, "roll": roll},
                            "prediction_result": {
                                "timestamp": ts,
                                "prediction": float(np.clip(pr*100.0,0.0,100.0)),
                                "confidence": conf
                            }
                        }
                        records.append(rec)

            if use_smooth_k and len(preds)>0:
                preds = majority_smooth(preds, k=int(use_smooth_k))

            if return_json:
                return records
            return (preds, probs) if return_prob else preds


if __name__=="__main__":
    username = "username"
    runner = PersonalizedModelRunner(username, use_personal=True)
