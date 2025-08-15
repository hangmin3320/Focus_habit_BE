import os, json, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CKPT_DIR = "app/models/checkpoint"

class TimeSeriesCNNGRU(nn.Module):
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
        x=x.permute(0,2,1)
        x=self.pool1(self.relu(self.bn1(self.conv1(x))))
        if self.two_convs:
            x=self.pool2(self.relu(self.bn2(self.conv2(x))))
        x=x.permute(0,2,1)
        _,h=self.gru(x)
        h=self.dropout(h[-1])
        return self.fc(h)

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns: df[c]=0.0
    return df

def feature_engineer(df, base=('ear','pitch','yaw','roll'), username="USER"):
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
    return max(1,int(window_size*(1-overlap)))

def create_sequences(df, features, window_size=25, stride=12, threshold=0.5):
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
    def __init__(self, patience=3, min_delta=0.0):
        self.patience=patience; self.min_delta=min_delta
        self.best=np.inf; self.count=0; self.stop=False
    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best=val_loss; self.count=0
        else:
            self.count+=1
            if self.count>=self.patience: self.stop=True

def _sweep_threshold(probs, labels, grid=None):
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

    # 검증 set 기준 정확도(0.5)도 참고용 기록
    y_val_np=y_val.astype(int)
    with torch.no_grad():
        lg_val=model(torch.tensor(X_val,dtype=torch.float32).to(device)).squeeze(-1)
        pr_val=(torch.sigmoid(lg_val).cpu().numpy()>=0.5).astype(int)
    val_acc=float((pr_val==y_val_np).mean())

    header={
        "username":username,"window_size":window_size,"dropout":0.3,"k1":3,"gru_hidden":32,
        "overlap":overlap,"stride":stride,"mode":mode,"epochs":epochs,"lr":lr,"wd":wd,
        "features":feats,"val_acc":val_acc,"best_val_loss":float(best_val),
        "threshold":best_t,"smooth_k":int(smooth_k) if (smooth_k and smooth_k>1) else None
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
