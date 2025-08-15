import os, json, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, threading

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

def majority_smooth(labels, k=3):
    if k is None or k<=1: return labels
    from collections import deque
    buf=deque(maxlen=k); out=[]
    for v in labels:
        buf.append(v)
        out.append(int(round(np.mean(buf))))
    return out

class PersonalizedModelRunner:
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
        if os.path.isfile(self.header_path):
            with open(self.header_path,'r',encoding='utf-8') as f:
                hdr=json.load(f)
            self.window_size=int(hdr.get("window_size",25))
            self.overlap=float(hdr.get("overlap",0.5))
            self.header_feats=hdr.get("features",None)
            self.threshold=float(hdr.get("threshold",0.5))
            self.smooth_k=hdr.get("smooth_k",None)

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

    def push_and_infer(self, json_payload, threshold=None, return_prob=False, smooth_k=None):
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
            preds=[]; probs=[]

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
                        pr=torch.sigmoid(self.model(x).squeeze(-1)).item()
                        probs.append(pr)
                        preds.append(1 if pr>=thr else 0)

            if use_smooth_k and len(preds)>0:
                preds = majority_smooth(preds, k=int(use_smooth_k))

            return (preds, probs) if return_prob else preds

if __name__=="__main__":
    username = "username"
    runner = PersonalizedModelRunner(username, use_personal=True)
