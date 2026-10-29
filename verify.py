"""Audit protocol artifacts and mathematical symmetry after a completed run."""
import json, subprocess
import numpy as np
import torch
from pypdf import PdfReader
from experiment import ROOT, ART, Forecaster, checks, transform, inverse, symmetrised

def main():
    checks()
    assert sum(p.numel() for p in Forecaster().parameters())==480784
    assert sum(p.numel() for p in Forecaster(True).parameters())==480784
    for n in 'ABCD':
        for s in [0,1]:
            h=json.loads((ART/f'train_{n}_seed{s}.json').read_text())
            assert len(h)==30 and [r['epoch'] for r in h]==list(range(1,31))
            assert all(np.isfinite(r['loss']) and r['seconds']>0 for r in h)
    for n in ['A','B','C','D','symB','symD']:
        for s in [0,1]:
            m=json.loads((ART/f'metrics_{n}_seed{s}.json').read_text())
            assert len(m['mse_h'])==5
            assert np.isclose(m['E5'],np.mean(m['mse_h']))
            assert np.isclose(m['E2_1'],np.mean(m['mse_h'][:2]))
            assert m['parameters']==480784
            if n.startswith('sym'):
                assert m['CM']<1e-12 and m['CR']<1e-12
                assert np.isclose(m['EM'],m['E5']) and np.isclose(m['ER'],m['E5'])
                base=json.loads((ART/f'metrics_{n[-1]}_seed{s}.json').read_text())
                assert base['training_seconds_per_epoch']==m['training_seconds_per_epoch']
    for n in ['Blank','Copy-last']:
        m=json.loads((ART/f'metrics_{n}.json').read_text()); assert m['CM']==0 and m['CR']==0
    for n in ['A','B','C','D','symB','symD']:
        p=torch.load(ART/f'predictions/{n}_seed0.pt',weights_only=True)
        assert all(tuple(t.shape)==(1000,5,1,64,64) for t in p['preds'].values())
    report=ROOT/'report'; text='\n'.join(p.extract_text() for p in PdfReader(report/'Final_Report.pdf').pages)
    for i in range(1,18): assert f'Q2.{i} ' in text
    for i in range(1,7): assert f'Table {i} ' in text and f'Figure {i}' in text
    assert len(list((report/'gifs').glob('*.gif')))==8
    for p in (report/'gifs').glob('*.gif'):
        from PIL import Image
        with Image.open(p) as im: assert im.n_frames==5
    log=subprocess.check_output(['git','log','--format=%s','--','predictions'],cwd=ROOT,text=True)
    assert 'Q2.5' in log and 'Q2.11' in log
    audit=dict(passed=True,checks=['exact transforms/windows','eight 30-epoch training histories','12 learned-model evaluations','1000-video prediction coverage','metric arithmetic','symmetry numerical precision','report answer/table/figure coverage','eight five-frame GIFs','prediction commits'],report_pages=len(PdfReader(report/'Final_Report.pdf').pages))
    (report/'verification.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
