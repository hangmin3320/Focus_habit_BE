import os, json, warnings, numpy as np, pandas as pd, torch, pickle
import torch.nn as nn
warnings.filterwarnings('ignore')

WIN=15; DROPOUT=0.3; K1=5; K2=5
TAG=f"win{WIN}_do{str(DROPOUT).replace('.','')}_k{K1}-{K2}"
CKPT_DIR="app/models/checkpoints"

def _ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns: df[c]=0.0
    return df

def feature_engineer(df, base=('ear','pitch','yaw','roll')):
    df=_ensure_cols(df,list(base)+['eye_status','prefix'])
    df=df.sort_values(['prefix','timestamp_ms'])
    for f in base:
        df[f]=df[f].fillna(0.0)
        df[f'{f}_diff']=df.groupby('prefix')[f].diff().fillna(0)
        m=df.groupby('prefix')[f].rolling(window=5,min_periods=1).mean().reset_index(level=0,drop=True)
        s=df.groupby('prefix')[f].rolling(window=5,min_periods=1).std().fillna(0).reset_index(level=0,drop=True)
        df[f'{f}_mean_5']=m; df[f'{f}_std_5']=s
    df['eye_status_numeric']=df['eye_status'].map({'OPEN':1,'CLOSED':0}).fillna(0)
    tmp=df.groupby('prefix')['eye_status_numeric'].diff().eq(-1)
    df['blink_count']=tmp.rolling(window=5,min_periods=1).sum().fillna(0).reset_index(level=0,drop=True)
    df['angle_magnitude']=np.sqrt(df['pitch_diff']**2+df['yaw_diff']**2+df['roll_diff']**2)
    feats=list(base)+[f'{f}_diff' for f in base]+[f'{f}_mean_5' for f in base]+[f'{f}_std_5' for f in base]+['blink_count','angle_magnitude']
    df[feats]=df[feats].replace([np.inf,-np.inf],0).fillna(0)
    return df,feats

def json_to_df(obj,prefix="STREAM"):
    if isinstance(obj,str):
        with open(obj,'r',encoding='utf-8') as f: arr=json.load(f)
    else:
        arr=obj
    rows=[]
    for it in arr:
        ts=it.get('timestamp',it.get('timestamp_ms',None))
        es=it.get('eye_status',{})
        hp=it.get('head_pose',{})
        ear_val=es.get('ear_value',es.get('ear',0.0))
        rows.append({'timestamp_ms':ts if ts is not None else np.nan,
                     'eye_status':es.get('status','OPEN'),
                     'ear':float(ear_val) if ear_val is not None else 0.0,
                     'pitch':float(hp.get('pitch',0.0)),
                     'yaw':float(hp.get('yaw',0.0)),
                     'roll':float(hp.get('roll',0.0)),
                     'label':0,'prefix':prefix})
    df=pd.DataFrame(rows)
    if 'timestamp_ms' not in df.columns or df['timestamp_ms'].isna().all():
        df['timestamp_ms']=np.arange(len(df))
    df=df[df['eye_status']!='NO_FACE_DETECTED'].sort_values('timestamp_ms')
    return df

class TimeSeriesCNN(nn.Module):
    def __init__(self,c,w,do=0.3,k1=5,k2=5):
        super().__init__()
        self.c1=nn.Conv1d(c,64,kernel_size=k1,padding=k1//2)
        self.r1=nn.ReLU(); self.p1=nn.MaxPool1d(2)
        self.c2=nn.Conv1d(64,32,kernel_size=k2,padding=k2//2)
        self.r2=nn.ReLU(); self.p2=nn.MaxPool1d(2)
        self.f=nn.Flatten()
        self.fc1=nn.Linear(32*(w//4),128)
        self.r3=nn.ReLU(); self.d=nn.Dropout(do)
        self.fc2=nn.Linear(128,1)
    def forward(self,x):
        x=x.permute(0,2,1)
        x=self.p1(self.r1(self.c1(x)))
        x=self.p2(self.r2(self.c2(x)))
        x=self.f(x)
        x=self.d(self.r3(self.fc1(x)))
        return self.fc2(x)

class StreamStandardizer:
    def __init__(self,eps=1e-6):
        self.n=0; self.mu=None; self.M2=None; self.eps=eps
    def partial_fit(self,X):
        X=np.asarray(X)
        if X.ndim==1: X=X[None,:]
        if self.mu is None:
            self.n=X.shape[0]
            self.mu=X.mean(axis=0)
            self.M2=((X-self.mu)**2).sum(axis=0)
        else:
            for x in X:
                self.n+=1
                d=x-self.mu
                self.mu+=d/self.n
                self.M2+=d*(x-self.mu)
    def transform(self,X):
        if self.mu is None:
            return X
        var=self.M2/np.maximum(self.n-1,1)
        std=np.sqrt(np.maximum(var,self.eps))
        return (X-self.mu)/std
    def fit_transform(self,X):
        self.partial_fit(X); return self.transform(X)

class FocusEngine:
    def __init__(self, ckpt_dir=CKPT_DIR, threshold=0.5, device=None):
        self.device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold=threshold
        self.buffer=pd.DataFrame(columns=['timestamp_ms','eye_status','prefix','label'])
        d0=pd.DataFrame([{'timestamp_ms':0,'eye_status':'OPEN','prefix':'D','label':0,'ear':0.0,'pitch':0.0,'yaw':0.0,'roll':0.0}])
        _,feats=feature_engineer(d0.copy()); self.features=feats

        scaler_path=os.path.join(ckpt_dir,f"scaler_{TAG}.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path,'rb') as f:
                self.std=pickle.load(f)
            self.use_stream_scaler=False
        else:
            self.std=StreamStandardizer()
            self.use_stream_scaler=True

        self.model=TimeSeriesCNN(len(self.features),WIN,DROPOUT,K1,K2).to(self.device)
        ckpts=[f"best_loss_cnn_{TAG}.pth",f"best_acc_cnn_{TAG}.pth",f"best_auc_cnn_{TAG}.pth",f"last_cnn_model_{TAG}.pth"]
        path=None
        for n in ckpts:
            p=os.path.join(ckpt_dir,n)
            if os.path.exists(p): path=p; break
        if path is None: raise FileNotFoundError("checkpoint not found in "+ckpt_dir)
        self.model.load_state_dict(torch.load(path,map_location=self.device))
        self.model.eval()
        self.last_emitted_start=None

    def _stride(self,win=WIN,stride=None,overlap_ratio=None):
        if stride is not None: return max(1,int(stride))
        if overlap_ratio is None: overlap_ratio=0.5
        return max(1,int(win*(1-overlap_ratio)))

    def _append(self,df):
        df=pd.concat([self.buffer,df],ignore_index=True)
        df,feats=feature_engineer(df)
        for c in self.features:
            if c not in df.columns: df[c]=0.0
        X=df[self.features].values
        if self.use_stream_scaler:
            if self.std.n==0 and len(X)>=WIN:
                self.std.partial_fit(X[:WIN])
            Xs=self.std.transform(X)
        else:
            Xs=self.std.transform(X)
        df[self.features]=Xs
        self.buffer=df

    def update_and_predict(self,incoming,prefix="STREAM",stride=None,overlap_ratio=None):
        if isinstance(incoming,(str,list,tuple)):
            inc=json_to_df(incoming,prefix=prefix)
        elif isinstance(incoming,pd.DataFrame):
            inc=incoming.copy()
        else:
            raise ValueError("invalid incoming type")
        self._append(inc)
        feats=self.buffer[self.features].values
        ts=self.buffer['timestamp_ms'].values
        n=len(self.buffer); win=WIN; st=self._stride(win,stride,overlap_ratio)
        out=[]
        if self.last_emitted_start is None:
            if n>=win:
                start=0
                x=torch.tensor(feats[start:start+win],dtype=torch.float32,device=self.device).unsqueeze(0)
                with torch.no_grad():
                    p=torch.sigmoid(self.model(x).squeeze(-1)).item()
                out.append({'window_start_idx':start,'window_end_idx':start+win-1,'timestamp_start':ts[start],'timestamp_end':ts[start+win-1],'prob':p,'pred':int(p>=self.threshold)})
                self.last_emitted_start=start
        if self.last_emitted_start is not None:
            start=self.last_emitted_start+st
            while start+win<=n:
                x=torch.tensor(feats[start:start+win],dtype=torch.float32,device=self.device).unsqueeze(0)
                with torch.no_grad():
                    p=torch.sigmoid(self.model(x).squeeze(-1)).item()
                out.append({'window_start_idx':start,'window_end_idx':start+win-1,'timestamp_start':ts[start],'timestamp_end':ts[start+win-1],'prob':p,'pred':int(p>=self.threshold)})
                self.last_emitted_start=start
                start=self.last_emitted_start+st
        return pd.DataFrame(out)

def predict_file(json_path, prefix="STREAM", overlap_ratio=0.5, stride=None, threshold=0.5, save_csv_path=None):
    eng=FocusEngine(threshold=threshold)
    df=json_to_df(json_path,prefix=prefix)
    outs=[]
    for i in range(len(df)):
        o=eng.update_and_predict(df.iloc[[i]],prefix=prefix,stride=stride,overlap_ratio=overlap_ratio)
        if len(o): outs.append(o)
    if len(outs)==0:
        res=pd.DataFrame(columns=['window_start_idx','window_end_idx','timestamp_start','timestamp_end','prob','pred'])
    else:
        res=pd.concat(outs,ignore_index=True)
    if save_csv_path: res.to_csv(save_csv_path,index=False)
    return res

if __name__=="__main__":
    pass
