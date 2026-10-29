"""Assignment 2: exact ConvLSTM protocol, two seeds, resumable experiments.

Run with CUDA_VISIBLE_DEVICES=2. Predictions are float32; no teacher forcing.
The test split is used for the prescribed evaluation only, never selection.
"""
from pathlib import Path
import json, os, random, time, hashlib, subprocess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
DEVICE = 'cuda'

def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)

def status(stage, **kw):
    save_json(ROOT/'status.json', dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw))
    print(stage, kw, flush=True)

def transform(x, g):
    """g=0..3: R^g; g=4..7: M R^(g-4)."""
    y = torch.rot90(x, g % 4, (-2,-1))
    return y.flip(-1) if g >= 4 else y

def inverse(x, g):
    return transform(x, g) if g >= 4 else transform(x, (-g) % 4)

def window(v, t=1, k=1):
    """1-based start; always five inputs, variable number of valid targets."""
    ix = t-1+torch.arange(5, device=v.device)*k
    iy = t-1+torch.arange(5,10, device=v.device)*k
    return v.index_select(-4, ix), v.index_select(-4, iy[iy < 20])

class Cell(nn.Module):
    def __init__(self, inc, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(inc+64,256,3,padding=dilation,dilation=dilation,bias=True)
    def forward(self, u, h, c):
        i,f,o,g = self.conv(torch.cat((u,h),1)).chunk(4,1)
        c = f.sigmoid()*c+i.sigmoid()*g.tanh()
        return o.sigmoid()*c.tanh(), c

class Forecaster(nn.Module):
    def __init__(self, larger=False):
        super().__init__()
        # D changes only dilation, from 1 to 2, retaining dense local input coverage
        # in layer 1 and extending layer 2. Longest radius = (1+2)*4=12 px.
        self.cells=nn.ModuleList([Cell(16,1), Cell(64,2 if larger else 1)])
        self.readout=nn.Conv2d(64,16,1,bias=True)
    def forward(self, x):
        b=x.shape[0]
        h1=x.new_zeros((b,64,16,16)); c1=torch.zeros_like(h1)
        h2=torch.zeros_like(h1); c2=torch.zeros_like(h1)
        out=[]; pred=None
        for step in range(9):
            u=x[:,step] if step < 5 else pred
            h1,c1=self.cells[0](F.pixel_unshuffle(u,4),h1,c1)
            h2,c2=self.cells[1](h1,h2,c2)
            pred=F.pixel_shuffle(self.readout(h2),4).sigmoid()
            if step >= 4: out.append(pred)
        return torch.stack(out,1)

def symmetrised(model, x):
    return torch.stack([inverse(model(transform(x,g)),g) for g in range(8)]).mean(0)

def model_fingerprint(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def checks():
    torch.manual_seed(0); x=torch.rand(20,1,64,64)
    p=torch.zeros_like(x); p[7,0,10,20]=1
    rows=[]
    r4=x
    for _ in range(4): r4=transform(r4,1)
    for name,a,b in [('a',transform(transform(x,4),4),x),('b',r4,x),('c',transform(transform(transform(x,4),1),4),transform(x,3))]:
        gap=float((a-b).abs().max()); eq=torch.equal(a,b)
        assert gap==0 and eq
        rows.append(dict(check=name,max_abs=gap,equal=eq))
    for name,g,expected in [('d',4,[7,10,43]),('e',1,[7,43,10])]:
        pos=torch.nonzero(transform(p,g))[0][[0,2,3]].tolist()
        assert pos==expected; rows.append(dict(check=name,position=pos))
    v=torch.arange(1,21).float()[:,None,None,None].expand(20,1,64,64)
    for name,k,ex,ey in [('f',1,[1,2,3,4,5],[6,7,8,9,10]),('g',2,[1,3,5,7,9],[11,13,15,17,19]),('h',3,[1,4,7,10,13],[16,19])]:
        a,b=window(v,k=k); aa=a[:,0,0,0].int().tolist(); bb=b[:,0,0,0].int().tolist()
        assert aa==ex and bb==ey; rows.append(dict(check=name,inputs=aa,targets=bb))
    for g in range(8): assert torch.equal(inverse(transform(x,g),g),x)
    assert sum(p.numel() for p in Forecaster().parameters()) == 480784
    assert sum(p.numel() for p in Forecaster(True).parameters()) <= 721176
    # A variable-horizon loss must average per video, rather than per valid frame.
    fake=torch.tensor([[1.,1.,0.,0.,0.],[4.,4.,4.,4.,4.]])
    mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,1]],dtype=torch.bool)
    assert float(((fake*mask).sum(1)/mask.sum(1)).mean())==2.5
    save_json(ART/'table1_checks.json',rows)

def dataset():
    from torchvision.datasets import MovingMNIST
    # Same official file and torchvision MD5; upgrade transport from HTTP to HTTPS.
    MovingMNIST._URL=MovingMNIST._URL.replace('http://','https://')
    d=MovingMNIST(ROOT/'data',split=None,download=True)
    assert tuple(d.data.shape)==(10000,20,1,64,64) and d.data.dtype==torch.uint8
    path=ROOT/'data/MovingMNIST/mnist_test_seq.npy'
    md5=hashlib.md5()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(1<<22),b''): md5.update(buf)
    assert md5.hexdigest()=='be083ec986bfe91a449d63653c411eb2'
    save_json(ART/'data_provenance.json',dict(md5=md5.hexdigest(),source=MovingMNIST._URL,shape=list(d.data.shape),train=[0,7999],validation=[8000,8999],test=[9000,9999]))
    return d.data

def optical_flow(videos):
    import cv2
    dirs=[]; speeds=[]; overlap_proxies=[]
    for v in videos[:50]:
        for s in range(19):
            a=v[s,0].numpy(); b=v[s+1,0].numpy()
            f=cv2.calcOpticalFlowFarneback(a,b,None,0.5,3,15,3,5,1.2,0)
            mag=np.sqrt((f*f).sum(-1)); mask=(a>51)&(b>51)&(mag>.5)
            dirs.append(np.degrees(np.arctan2(-f[...,1][mask],f[...,0][mask]))%360)
            speeds.append(mag[mask])
    dirs=np.concatenate(dirs); speeds=np.concatenate(speeds)
    np.savez_compressed(ART/'flow.npz',direction=dirs,speed=speeds)
    bins=np.histogram(dirs,bins=np.arange(0,361,20))[0]/len(dirs)
    save_json(ART/'flow_summary.json',dict(pairs=950,pixels=len(speeds),median_speed=float(np.median(speeds)),direction_fractions=bins.tolist()))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def sample_batch(videos, indexes, name, rng):
    # Single RNG controls windows and augmentation independent of GPU kernels.
    b=len(indexes); xs=[]; ys=[]; masks=[]
    for ix in indexes.tolist():
        k=int(rng.integers(1,4)) if name in ('C','D') else 1
        maxstart={1:14,2:8,3:2}[k] if name in ('C','D') else 11
        t=int(rng.integers(1,maxstart+1)); x,y=window(videos[ix],t,k)
        if name=='B':
            g=int(rng.integers(0,8)); x=transform(x,g); y=transform(y,g)
        n=y.shape[0]; ys.append(F.pad(y,(0,0,0,0,0,0,0,5-n)))
        xs.append(x); masks.append(torch.arange(5)<n)
    return (torch.stack(xs).to(DEVICE,dtype=torch.float32).div_(255),
            torch.stack(ys).to(DEVICE,dtype=torch.float32).div_(255),
            torch.stack(masks).to(DEVICE))

def train(videos,name,seed):
    path=ROOT/'checkpoints'/f'{name}_seed{seed}.pt'; path.parent.mkdir(exist_ok=True)
    seed_all(seed); rng=np.random.default_rng(seed)
    model=Forecaster(name=='D').to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    start=0; history=[]
    if path.exists():
        ck=torch.load(path,map_location=DEVICE,weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        start=ck['epoch']; history=ck['history']; rng.bit_generator.state=ck['rng']
        torch.set_rng_state(ck['torch_rng'].cpu()); torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
    # Float32 arithmetic and fixed protocol, no AMP, schedule or early stopping.
    for ep in range(start,30):
        model.train(); torch.cuda.synchronize(); tic=time.perf_counter(); losses=[]
        for indexes in np.array_split(rng.permutation(8000),250):
            x,y,m=sample_batch(videos,indexes,name,rng)
            opt.zero_grad(set_to_none=True); pred=model(x)
            perframe=(pred-y).square().mean((2,3,4))
            loss=((perframe*m).sum(1)/m.sum(1)).mean()
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        torch.cuda.synchronize(); elapsed=time.perf_counter()-tic
        history.append(dict(epoch=ep+1,loss=float(np.mean(losses)),seconds=elapsed))
        tmp=path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep+1,history=history,rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),name=name,seed=seed),tmp); tmp.replace(path)
        save_json(ART/f'train_{name}_seed{seed}.json',history)
        status('training',model=name,seed=seed,epoch=ep+1,loss=history[-1]['loss'],epoch_seconds=elapsed)
    return model.eval(),float(np.mean([r['seconds'] for r in history]))

@torch.no_grad()
def predict(model,test,k=1,g=0,wrapped=False):
    out=[]; timing=0.
    # Warm-up outside timed batches. Inference timer excludes transfer and data conversion.
    x,_=window(test[:32].to(DEVICE,dtype=torch.float32)/255,k=k)
    (symmetrised(model,x) if wrapped else model(x)); torch.cuda.synchronize()
    for raw in test.split(32):
        v=raw.to(DEVICE,dtype=torch.float32)/255; x,_=window(v,k=k)
        x=transform(x,g); torch.cuda.synchronize(); tic=time.perf_counter()
        pred=symmetrised(model,x) if wrapped else model(x)
        torch.cuda.synchronize(); timing+=time.perf_counter()-tic
        out.append(pred.cpu())
    return torch.cat(out),timing

def score(pred,target):
    return (pred[:,:target.shape[1]]-target).square().mean((2,3,4))

def summarise(preds,test):
    v=test.float()/255
    _,y1=window(v); _,y2=window(v,k=2); _,y3=window(v,k=3)
    e1=score(preds['1'],y1); e2=score(preds['2'],y2); e3=score(preds['3'],y3)
    cm=(preds['M']-transform(preds['1'],4)).square().mean((1,2,3,4))
    cr=(preds['R']-transform(preds['1'],1)).square().mean((1,2,3,4))
    em=score(preds['M'],transform(y1,4)).mean(); er=score(preds['R'],transform(y1,1)).mean()
    metrics=dict(mse_h=e1.mean(0).tolist(),E5=float(e1.mean()),CM=float(cm.mean()),EM=float(em),CR=float(cr.mean()),ER=float(er),E2_1=float(e1[:,:2].mean()),E2_2=float(e2[:,:2].mean()),E2_3=float(e3[:,:2].mean()),E5_2=float(e2.mean()))
    return metrics,cm,cr

def evaluate(model,test,name,seed,train_time,wrapped=False):
    prefix=f'{name}_seed{seed}'; marker=ART/f'metrics_{prefix}.json'
    fingerprint=model_fingerprint(model)
    if marker.exists():
        cached=json.loads(marker.read_text())
        if cached.get('model_sha256')==fingerprint and (seed!=0 or (ART/f'predictions/{prefix}.pt').exists()):
            return cached
    status('evaluation',model=name,seed=seed)
    preds={}; infer=0
    for key,k,g in [('1',1,0),('M',1,4),('R',1,1),('2',2,0),('3',3,0)]:
        preds[key],sec=predict(model,test,k,g,wrapped)
        if key=='1': infer=sec
    met,cm,cr=summarise(preds,test)
    met.update(parameters=sum(p.numel() for p in model.parameters()),training_seconds_per_epoch=train_time,inference_seconds=infer,model_sha256=fingerprint)
    out=ART/'predictions'; out.mkdir(exist_ok=True)
    if seed==0:
        torch.save(dict(preds=preds,CM=cm,CR=cr),out/f'{prefix}.pt')
    save_json(marker,met); return met

def references(test):
    v=test.float()/255
    for name in ['Blank','Copy-last']:
        preds={}
        for key,k,g in [('1',1,0),('M',1,4),('R',1,1),('2',2,0),('3',3,0)]:
            x,_=window(v,k=k); x=transform(x,g)
            preds[key]=torch.zeros((len(v),5,1,64,64)) if name=='Blank' else x[:,-1:].expand(-1,5,-1,-1,-1)
        met,_,_=summarise(preds,test); met['parameters']=0
        save_json(ART/f'metrics_{name}.json',met)

def commit_prediction(number,text):
    p=ROOT/'predictions'/f'Q2_{number}.md'; p.parent.mkdir(exist_ok=True)
    if not p.exists():
        p.write_text(text+'\n')
        subprocess.run(['git','add',str(p.relative_to(ROOT))],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m',f'Record Q2.{number} prediction before corresponding experiments'],cwd=ROOT,check=True)

def main():
    import fcntl
    lock=(ROOT/'.run.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ART.mkdir(exist_ok=True); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    status('setup'); checks()
    save_json(ART/'hardware.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),threads=torch.get_num_threads(),precision='float32; TF32 disabled',cudnn_benchmark=True))
    videos=dataset()
    if not (ART/'flow_summary.json').exists(): optical_flow(videos)
    # Test videos are not used to choose any architecture or hyperparameter.
    test=videos[9000:10000]
    models=[]
    for seed in [0,1]:
        model,tt=train(videos,'A',seed); models.append((model,tt))
    flow=json.loads((ART/'flow_summary.json').read_text()); median=flow['median_speed']
    commit_prediction('5',f'I predict model A will be approximately, but not exactly, mirror-consistent; rotation will be harder because digit appearance changes more. Its longest one-step centre-to-centre reach is 8 original pixels, whereas the measured typical speeds are {median:.3g}, {2*median:.3g}, and {3*median:.3g} pixels per observed frame at 1x, 2x, and 3x. I expect quality to deteriorate at 2x and more strongly at 3x, especially for individual displacements above 8 pixels; reach alone is not a hard accuracy threshold.')
    references(test)
    for seed,(model,tt) in enumerate(models): evaluate(model,test,'A',seed,tt)
    del models,model; torch.cuda.empty_cache()
    commit_prediction('11','I predict B will reduce mirror and rotation inconsistency and transformed error, but may sacrifice some original-clip accuracy. C should improve fast-motion forecasts by exposing the baseline to speeds 2x and 3x, while symmetry remains unconstrained. D increases one-step reach from 8 to 12 pixels and should improve over C for larger displacements, although bounces, overlaps, and optimisation may prevent a complete fix.')
    for name in ['B','C','D']:
        for seed in [0,1]:
            model,tt=train(videos,name,seed)
            evaluate(model,test,name,seed,tt)
            if name in ('B','D'): evaluate(model,test,'sym'+name,seed,tt,wrapped=True)
            del model; torch.cuda.empty_cache()
    status('rendering_report')
    from reporting import build
    build()
    from verify import main as verify_outputs
    verify_outputs()
    status('complete',report=str(ROOT/'report/Final_Report.pdf'))
    subprocess.run(['git','add','experiment.py','reporting.py','README.md','requirements.txt','requirements-lock.txt','report','artifacts','predictions','.gitignore','verify.py','package_results.py','status.json'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','Complete two-seed experiments, analysis, figures and final report'],cwd=ROOT,check=False)
    subprocess.run(['git','bundle','create',str(ROOT/'project_history.bundle'),'--all'],cwd=ROOT,check=True)
    from package_results import main as package_outputs
    package_outputs()

if __name__=='__main__':
    try: main()
    except Exception as exc:
        status('failed',error=repr(exc)); raise
