"""Make a compact, self-contained deliverable without dataset/prediction caches."""
from pathlib import Path
import csv, hashlib, json, zipfile
import torch
from experiment import ROOT, ART

def main():
    report=ROOT/'report'; weights=ROOT/'model_weights'; weights.mkdir(exist_ok=True)
    for n in 'ABCD':
        for s in [0,1]:
            ck=torch.load(ROOT/f'checkpoints/{n}_seed{s}.pt',map_location='cpu',weights_only=False)
            assert ck['epoch']==30
            torch.save(dict(state_dict=ck['model'],model=n,seed=s,epochs=30,larger=n=='D',reach_pixels_per_coordinate=12 if n=='D' else 8),weights/f'{n}_seed{s}.pt')
    rows=[]
    for n in ['Blank','Copy-last','A','B','C','D','symB','symD']:
        for seed in ([None] if n in ['Blank','Copy-last'] else [0,1]):
            suffix=f'_seed{seed}' if seed is not None else ''
            m=json.loads((ART/f'metrics_{n}{suffix}.json').read_text())
            row=dict(model=n,seed=seed)
            row.update({k:v for k,v in m.items() if k!='mse_h'})
            row.update({f'MSE_{i+1}':v for i,v in enumerate(m['mse_h'])})
            rows.append(row)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (report/'all_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    paths=[]
    for name in ['experiment.py','reporting.py','verify.py','package_results.py','README.md','requirements.txt','requirements-lock.txt','status.json']:
        paths.append(ROOT/name)
    for folder in [report,weights,ROOT/'predictions']:
        paths.extend(p for p in folder.rglob('*') if p.is_file() and not p.name.startswith('preview_'))
    paths.extend(ART.glob('*.json'))
    paths.append(ART/'flow.npz')
    paths.extend(p for p in (ROOT/'Question1').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.pt','.json','.ipynb','.txt'])
    # Include local Git prediction evidence as plain text, not private Git metadata.
    destination=ROOT/'DL4STG_PA2_Completed.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(paths)): z.write(p,p.relative_to(ROOT))
    sha=hashlib.sha256(destination.read_bytes()).hexdigest()
    (ROOT/'DL4STG_PA2_Completed.zip.sha256').write_text(sha+'  '+destination.name+'\n')
    print(f'Packaged {len(set(paths))} files: {destination} ({destination.stat().st_size/1e6:.2f} MB)')

if __name__=='__main__': main()
