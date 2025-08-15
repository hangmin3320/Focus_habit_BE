import os, json, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CKPT_DIR = "app/models/checkpoint"

class TimeSeriesCNNGRU(nn.Module):
    """
    1D-CNN + GRU 기반 시계열 이진 분류 모델.

    Args:
        input_channels (int): 입력 채널 수(피처 개수).
        window_size (int): 시퀀스 길이(윈도우 크기).
        dropout_rate (float): 드롭아웃 확률.
        k1 (int): 첫 번째 컨볼루션 커널 크기.
        k2 (int): GRU hidden size.
        two_convs (bool): 2번째 컨볼루션 사용 여부.

    Shape:
        - input: (B, T, C)  where T=window_size, C=input_channels
        - output: (B, 1)    (logit)

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
            x (torch.Tensor): (B, T, C) 형태의 입력.

        Returns:
            torch.Tensor: (B, 1) 형태의 로짓(logit).
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
    DataFrame에 지정 컬럼이 없으면 0.0으로 생성.

    Args:
        df (pd.DataFrame): 입력 프레임.
        cols (list[str]): 보장하고 싶은 컬럼 목록.

    Returns:
        pd.DataFrame: 컬럼이 보장된 데이터프레임.
    """
    for c in cols:
        if c not in df.columns: df[c]=0.0
    return df


def feature_engineer(df, base=('ear','pitch','yaw','roll'), username="USER"):
    """
    시계열 파생 피처 생성 및 정렬/결측치 처리.

    생성 피처:
        - 각 base 피처의 1차 차분(_diff)
        - 이동평균 5(_mean_5), 이동표준편차 5(_std_5)
        - 눈깜빡임 개수(blink_count; 최근 5 윈도우)
        - 회전 변화량 벡터 크기(angle_magnitude)

    Args:
        df (pd.DataFrame): 원본 데이터프레임.
        base (tuple[str]): 기본 피처명 튜플.
        username (str): prefix 기본값.

    Returns:
        Tuple[pd.DataFrame, list[str]]: (가공된 DF, 학습용 피처 리스트)
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
    윈도우 겹침 비율에 따른 stride 계산.

    Args:
        window_size (int): 윈도우 길이.
        overlap (float): 겹침 비율(0~1).

    Returns:
        int: stride 값(최소 1).
    """
    return max(1,int(window_size*(1-overlap)))


def create_sequences(df, features, window_size=25, stride=12, threshold=0.5):
    """
    슬라이딩 윈도우로 (X, y) 시퀀스 생성.

    레이블링:
        - df['label']가 존재하면 윈도우 평균 >= threshold → 1, 아니면 0.
        - df['label']가 없으면 y는 빈 배열.

    Args:
        df (pd.DataFrame): 입력 데이터프레임(정렬 완료).
        features (list[str]): 사용 피처 목록.
        window_size (int): 시퀀스 길이.
        stride (int): 슬라이딩 보폭.
        threshold (float): 다수결/평균 임계값.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            X: (N, window_size, C)
            y: (N,) 이진 레이블(없으면 길이 0)
    """
    X,y=[],[]
    for p in df['prefix'].unique():
        g=df[df['prefix']==p].sort_values('timestamp_ms')
        if len(g)<window_size: continue
        arr=g[features].values
        lab=g['label'].values if 'label' in g.columns else None
        for i in range(0,len(g)-window_size+1,stride):
            X.append(arr[i:i+window_size])
            if lab is not None:
                seq=lab[i:i+window_size]
                y.append(1 if np.mean(seq)>=threshold else 0)
    X=np.array(X) if len(X) else np.empty((0,len(features),window_size))
    y=np.array(y) if len(y) else np.array([])
    return X,y


def json_to_df(path_or_obj, label=None, username="USER"):
    """
    JSON(list[dict])을 표준 DF로 변환.

    입력 JSON 예:
        {
          "timestamp": 1755199738824,
          "eye_status": {"status":"OPEN","ear_value":0.456},
          "head_pose": {"pitch":2.5,"yaw":60.0,"roll":160.0}
        }

    Args:
        path_or_obj (str|list|dict): 파일 경로, JSON 문자열, 혹은 파이썬 객체.
        label (int|None): 레이블(0/1). None이면 NaN 저장.
        username (str): prefix 값.

    Returns:
        pd.DataFrame: 표준 스키마의 데이터프레임.
    """
    if isinstance(path_or_obj,str) and os.path.isfile(path_or_obj):
        with open(path_or_obj,'r',encoding='utf-8') as f: data=json.load(f)
    elif isinstance(path_or_obj,(str,bytes)):
        data=json.loads(path_or_obj)
    else:
        data=path_or_obj
    rows=[]
    for it in data:
        ts=it.get('timestamp',it.get('timestamp_ms',None))
        es=it.get('eye_status',{})
        hp=it.get('head_pose',{})
        ear_val=es.get('ear_value',es.get('ear',0.0))
        rows.append({'timestamp_ms':ts if ts is not None else np.nan,'eye_status':es.get('status','OPEN'),
                     'ear':float(ear_val) if ear_val is not None else 0.0,
                     'pitch':float(hp.get('pitch',0.0)),'yaw':float(hp.get('yaw',0.0)),'roll':float(hp.get('roll',0.0)),
                     'prefix':username,'label':label if label is not None else np.nan})
    df=pd.DataFrame(rows)
    if 'timestamp_ms' not in df.columns: df['timestamp_ms']=np.arange(len(df))
    return df


def set_trainable(model, mode):
    """
    전이학습/개인화 단계에서 학습 가능한 파라미터 설정.

    Args:
        model (nn.Module): 학습 대상 모델.
        mode (str): 'head' | 'gru' | 'cnn_gru' | 'full'

    Returns:
        None
    """
    for p in model.parameters(): p.requires_grad=False
    if mode=='head':
        for p in model.fc.parameters(): p.requires_grad=True
    elif mode=='gru':
        for p in model.gru.parameters(): p.requires_grad=True
        for p in model.fc.parameters(): p.requires_grad=True
    elif mode=='cnn_gru':
        for p in model.conv1.parameters(): p.requires_grad=True
        for p in model.gru.parameters(): p.requires_grad=True
        for p in model.fc.parameters(): p.requires_grad=True
    elif mode=='full':
        for p in model.parameters(): p.requires_grad=True


class EarlyStopping:
    """
    검증 손실 기반 조기 종료.

    Args:
        patience (int): 개선 없는 에폭 허용 횟수.
        min_delta (float): 개선으로 인정할 최소 손실 감소폭.

    Attributes:
        best (float): 관측된 최소 검증 손실.
        count (int): 연속 비개선 횟수.
        stop (bool): 중지 여부 플래그.
    """

    def __init__(self, patience=3, min_delta=0.0):
        self.patience=patience; self.min_delta=min_delta
        self.best=np.inf; self.count=0; self.stop=False

    def step(self, val_loss):
        """
        한 에폭의 검증 손실을 반영하여 상태 업데이트.

        Args:
            val_loss (float): 현재 에폭의 검증 손실.

        Returns:
            None
        """
        if val_loss < self.best - self.min_delta:
            self.best=val_loss; self.count=0
        else:
            self.count+=1
            if self.count>=self.patience: self.stop=True


def _sweep_threshold(probs, labels, grid=None):
    """
    확률 예측(probs)에 대해 단순 정확도를 최대화하는 이진 임계값 탐색.

    Args:
        probs (np.ndarray): 예측 확률 (N,).
        labels (np.ndarray): 정답 레이블 (N,), {0,1}.
        grid (Iterable[float]|None): 탐색 임계값 리스트. None이면 0.1~0.9(0.01 step).

    Returns:
        Tuple[float, float]: (best_threshold, best_accuracy)
    """
    if grid is None: grid=np.linspace(0.1,0.9,81)
    best_t=0.5; best_acc=-1.0
    for t in grid:
        acc=((probs>=t).astype(int)==labels).mean()
        if acc>best_acc:
            best_acc=acc; best_t=t
    return float(best_t), float(best_acc)


def train_personal(username, json_focus_path, json_nonfocus_path,
                   base_model_path=None, base_scaler_path=None,
                   mode='gru', epochs=5, lr=1e-4, wd=1e-5,
                   window_size=25, overlap=0.5, smooth_k=3, thr_grid=None):
    """
    사용자 개인화 파인튜닝 파이프라인.

    단계:
        1) JSON → DF 변환 및 전처리/파생
        2) 스케일링 적합 및 적용(개인 스케일러)
        3) 슬라이딩 윈도우로 (X, y) 시퀀스 생성
        4) 사전학습 가중치 로드 후 지정 모드로 파라미터 고정/해제
        5) BCEWithLogitsLoss(+class weight)로 학습/검증 및 EarlyStopping
        6) 전체 데이터 확률로 best threshold 스윕
        7) 개인 모델/스케일러/헤더 저장

    Args:
        username (str): 사용자 ID.
        json_focus_path (str): 집중(json) 경로.
        json_nonfocus_path (str): 비집중(json) 경로.
        base_model_path (str|None): 베이스 모델 경로(없으면 기본 경로).
        base_scaler_path (str|None): 베이스 스케일러 경로(형식 검증용).
        mode (str): set_trainable 모드. {'head','gru','cnn_gru','full'}.
        epochs (int): 학습 에폭 수.
        lr (float): 학습률.
        wd (float): weight decay.
        window_size (int): 윈도우 길이.
        overlap (float): 윈도우 겹침 비율.
        smooth_k (int): 추론시 다수결 스무딩 길이(헤더 저장용).
        thr_grid (Iterable[float]|None): 임계값 스윕 그리드.

    Returns:
        dict|None: 저장 경로와 지표를 담은 dict. 데이터 부족 시 None.
    """
    os.makedirs(CKPT_DIR,exist_ok=True)
    if base_model_path is None: base_model_path=os.path.join(CKPT_DIR,"model.pth")
    if base_scaler_path is None: base_scaler_path=os.path.join(CKPT_DIR,"scaler.pkl")
    with open(base_scaler_path,'rb') as f: _=pickle.load(f)

    df_f=json_to_df(json_focus_path,label=0,username=username)
    df_n=json_to_df(json_nonfocus_path,label=1,username=username)
    df=pd.concat([df_f,df_n],ignore_index=True)
    if 'eye_status' in df.columns:
        df=df[df['eye_status']!='NO_FACE_DETECTED']

    df,feats=feature_engineer(df,username=username)
    scaler=StandardScaler()
    df[feats]=scaler.fit_transform(df[feats])

    stride=compute_stride(window_size,overlap)
    X,y=create_sequences(df,feats,window_size,stride,threshold=0.5)
    if X.shape[0]==0 or y.shape[0]==0:
        return None

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=TimeSeriesCNNGRU(len(feats),window_size,0.3,3,32,True).to(device)
    model.load_state_dict(torch.load(base_model_path,map_location=device))
    set_trainable(model,mode)

    if len(np.unique(y))>1:
        X_tr,X_val,y_tr,y_val=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    else:
        X_tr,X_val,y_tr,y_val=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)

    pos=max(1,int((y_tr==1).sum()))
    neg=max(1,int((y_tr==0).sum()))
    pos_weight=torch.tensor([neg/pos],device=device,dtype=torch.float32)

    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer=optim.Adam((p for p in model.parameters() if p.requires_grad),lr=lr,weight_decay=wd)

    train_ds=torch.utils.data.TensorDataset(torch.tensor(X_tr,dtype=torch.float32),torch.tensor(y_tr,dtype=torch.float32))
    val_ds=torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.float32),torch.tensor(y_val,dtype=torch.float32))
    train_dl=torch.utils.data.DataLoader(train_ds,batch_size=32,shuffle=True)
    val_dl=torch.utils.data.DataLoader(val_ds,batch_size=64,shuffle=False)

    early=EarlyStopping(patience=3,min_delta=0.0)
    best_sd=None; best_val=np.inf

    for _ in range(epochs):
        model.train()
        for xb,yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            lg=model(xb).squeeze(-1)
            loss=criterion(lg,yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()

        model.eval(); vloss=0.0; n=0
        with torch.no_grad():
            for xb,yb in val_dl:
                xb,yb=xb.to(device),yb.to(device)
                lg=model(xb).squeeze(-1)
                ls=criterion(lg,yb)
                vloss+=ls.item()*xb.size(0); n+=xb.size(0)
        vloss/=max(1,n)
        if vloss<best_val:
            best_val=vloss; best_sd=model.state_dict()
        if early.step(vloss) or np.isnan(vloss): break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    # 개인 임계값 학습 (focus/nonfocus 전체 데이터 확률로 스윕)
    with torch.no_grad():
        lg=model(torch.tensor(X,dtype=torch.float32).to(device)).squeeze(-1)
        probs=torch.sigmoid(lg).cpu().numpy()
    best_t, best_acc = _sweep_threshold(probs, y, grid=thr_grid)

    # 저장
    user_model_path=os.path.join(CKPT_DIR,f"{username}_model.pth")
    user_scaler_path=os.path.join(CKPT_DIR,f"{username}_scaler.pkl")
    torch.save(model.state_dict(),user_model_path)
    with open(user_scaler_path,'wb') as f: pickle.dump(scaler,f)

    # 검증 set 기준 정확도(0.5) 참고용
    y_val_np=y_val.astype(int)
    with torch.no_grad():
        lg_val=model(torch.tensor(X_val,dtype=torch.float32).to(device)).squeeze(-1)
        pr_val=(torch.sigmoid(lg_val).cpu().numpy()>=0.5).astype(int)
    val_acc=float((pr_val==y_val_np).mean())

    header={
        "username":username,"window_size":window_size,"dropout":0.3,"k1":3,"gru_hidden":32,
        "overlap":overlap,"stride":stride,"mode":mode,"epochs":epochs,"lr":lr,"wd":wd,
        "features":feats,"val_acc":val_acc,"best_val_loss":float(best_val),
        "threshold":best_t,"smooth_k":int(smooth_k) if (smooth_k and smooth_k>1) else None,
        "temperature": 1.0
    }
    with open(os.path.join(CKPT_DIR,f"{username}_header.json"),'w',encoding='utf-8') as f: json.dump(header,f,ensure_ascii=False)

    return {"model":user_model_path,"scaler":user_scaler_path,"header":os.path.join(CKPT_DIR,f"{username}_header.json"),
            "val_acc":val_acc,"best_threshold":best_t,"thr_sweep_acc":best_acc}


if __name__=="__main__":
    username = "username"
    data_dir = "app/data"

    # 파일명 규칙: app/data/{username}_focus.json, app/data/{username}_nonfocus.json
    focus_json    = os.path.join(data_dir, f"{username}_focus.json")
    nonfocus_json = os.path.join(data_dir, f"{username}_nonfocus.json")

    base_model  = os.path.join(CKPT_DIR, "model.pth")
    base_scaler = os.path.join(CKPT_DIR, "scaler.pkl")

    out = train_personal(
        username, focus_json, nonfocus_json,
        base_model, base_scaler,
        mode='gru', epochs=5, lr=1e-4, wd=1e-5,
        window_size=25, overlap=0.5, smooth_k=3, thr_grid=np.linspace(0.1, 0.9, 81)
    )
    print(out)
