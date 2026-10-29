"""Create numbered figures, GIFs, measured analysis, and a standalone PDF."""
from pathlib import Path
import json, math, io, html, subprocess
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from experiment import ROOT, ART, transform, window

REPORT=ROOT/'report'; FIG=REPORT/'figures'; GIF=REPORT/'gifs'
plt.rcParams.update({'font.size':9,'figure.dpi':150,'savefig.bbox':'tight'})

def load(name,seed=None):
    suffix=f'_seed{seed}' if seed is not None else ''
    return json.loads((ART/f'metrics_{name}{suffix}.json').read_text())

def fmt(v, scientific=False):
    if v==0: return '0.00e+00' if scientific else '0'
    return f'{v:.2e}' if scientific or abs(v)<1e-3 else f'{v:.3g}'

def pair(name,key,scientific=False,index=None):
    vals=[load(name,s)[key] for s in [0,1]]
    if key=='parameters':
        assert vals[0]==vals[1]
        return f'{vals[0]:,}'
    if index is not None: vals=[v[index] for v in vals]
    return f'{fmt(sum(vals)/2,scientific)} ({fmt(vals[0],scientific)}, {fmt(vals[1],scientific)})'

def mean(name,key): return sum(load(name,s)[key] for s in [0,1])/2

def grid(rows,labels,path,title='',col_labels=None):
    fig,ax=plt.subplots(len(rows),len(rows[0]),figsize=(12,len(rows)*1.3),squeeze=False)
    for r,row in enumerate(rows):
        for c,img in enumerate(row):
            ax[r,c].imshow(np.asarray(img).squeeze(),cmap='gray',vmin=0,vmax=1)
            ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
            for sp in ax[r,c].spines.values(): sp.set_visible(False)
            if c==0: ax[r,c].set_ylabel(labels[r],fontsize=8)
            if r==0 and col_labels: ax[r,c].set_title(col_labels[c],fontsize=8)
    fig.suptitle(title); fig.subplots_adjust(wspace=.02,hspace=.1,top=.88)
    fig.savefig(path); plt.close(fig)

def gif_panel(x,p,ty,tp,orig,path):
    frames=[]
    for h in range(5):
        rows=[torch.cat((x,p[h:h+1])),torch.cat((ty[:5],ty[5+h:6+h])),torch.cat((ty[:5],tp[h:h+1])),torch.cat((ty[:5],orig[h:h+1]))]
        fig,ax=plt.subplots(4,6,figsize=(9,5),squeeze=False)
        for r in range(4):
            for c in range(6):
                ax[r,c].imshow(rows[r][c].squeeze(),cmap='gray',vmin=0,vmax=1); ax[r,c].axis('off')
                if r==0: ax[r,c].set_title(f'input {c+1}' if c<5 else f'h={h+1}')
        fig.suptitle('Rows: original prediction / transformed truth / transformed input prediction / transformed original prediction',fontsize=9)
        buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=90); plt.close(fig); buf.seek(0)
        frames.append(PILImage.open(buf).convert('RGB'))
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=650,loop=0)

def component_count(frame):
    import cv2
    a=(np.asarray(frame).squeeze()>.2).astype(np.uint8)
    n,labels,stats,cents=cv2.connectedComponentsWithStats(a,8)
    return int((stats[1:,cv2.CC_STAT_AREA]>20).sum())

def separated_digits(frame):
    """Conservative example screen: two components with separated bounding boxes."""
    import cv2
    a=(np.asarray(frame).squeeze()>.05).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(a,8)
    boxes=[s for s in stats[1:] if s[cv2.CC_STAT_AREA]>12]
    if len(boxes)!=2: return False
    x1,y1,w1,h1,_=boxes[0]; x2,y2,w2,h2,_=boxes[1]
    return max(x2-(x1+w1),x1-(x2+w2),y2-(y1+h1),y1-(y2+h2))>=3

def make_figures(test):
    FIG.mkdir(parents=True,exist_ok=True); GIF.mkdir(exist_ok=True)
    flow=np.load(ART/'flow.npz'); dirs=flow['direction']; speeds=flow['speed']
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    ax[0].hist(dirs,bins=np.arange(0,361,20),weights=np.ones(len(dirs))/len(dirs),edgecolor='white')
    ax[0].axhline(1/18,color='crimson',ls='--',label='uniform 1/18'); ax[0].legend(); ax[0].set(xlabel='direction (degrees; 90 = up)',ylabel='fraction of kept pixels',title='(a) Direction')
    ax[1].hist(speeds,bins=35,weights=np.ones(len(speeds))/len(speeds),edgecolor='white')
    ax[1].axvline(np.median(speeds),color='crimson',ls='--',label=f'median {np.median(speeds):.3g}'); ax[1].legend(); ax[1].set(xlabel='pixels per frame',ylabel='fraction of kept pixels',title='(b) Speed')
    fig.tight_layout(); fig.savefig(FIG/'Figure1.png'); plt.close(fig)
    a=torch.load(ART/'predictions/A_seed0.pt',weights_only=True)
    final=torch.load(ART/'predictions/symD_seed0.pt',weights_only=True)
    v=test.float()/255; x,y=window(v)
    # An observed merger and later separation is a reproducible overlap candidate.
    overlaps=[i for i in range(1000) if 1 in [component_count(f) for f in v[i,:10]] and 2 in [component_count(f) for f in v[i,:10]]]
    example=[0,1,next((i for i in overlaps if i not in [0,1]),overlaps[0])]
    rows=[]; labels=[]
    for i in example:
        rows.extend([torch.cat((x[i],a['preds']['1'][i])),torch.cat((x[i],y[i]))]); labels.extend([f'video {9000+i}\nA forecast','truth'])
    grid(rows,labels,FIG/'Figure2.png','Figure 2 — Original 1x examples (seed 0)',[f'x{j+1}' for j in range(5)]+[f'h{j+1}' for j in range(5)])
    ids={g:torch.topk(a['C'+g],2).indices.tolist() for g in ['M','R']}
    for label,predobj,number in [('A',a,'3'),('symD',final,'6a')]:
        for g in ['M','R']:
            gi=4 if g=='M' else 1
            for rank,i in enumerate(ids[g],1):
                px=predobj['preds']['1'][i]; tp=predobj['preds'][g][i]
                tx=transform(x[i],gi); ty=transform(y[i],gi); orig=transform(px,gi)
                rows=[torch.cat((x[i],px)),torch.cat((tx,ty)),torch.cat((tx,tp)),torch.cat((tx,orig))]
                grid(rows,['X / f(X)','T(X) / T(Y)','T(X) / f(TX)','T(X) / T(fX)'],FIG/f'Figure{number}_{g}_{rank}.png',f'Figure {number} — {label}, {g}, dataset video {9000+i}',[f'input {j+1}' for j in range(5)]+[f'h={j+1}' for j in range(5)])
                gif_panel(x[i],px,torch.cat((tx,ty)),tp,orig,GIF/f'Figure{number}_{g}_{rank}.gif')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    for j,g in enumerate(['M','R']):
        vals=a['C'+g].numpy(); ax[j].hist(vals,bins=35,color='#337ea9',alpha=.8)
        for i in ids[g]: ax[j].axvline(vals[i],ls='--',label=f'video {9000+i}')
        ax[j].set(xlabel=f'per-video C_{g}',ylabel='number of test videos',title=g); ax[j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG/'Figure4.png'); plt.close(fig)
    display_indices=sorted(set([j for k in [1,2,3] for j in list(range(0,5*k,k))+[5*k,6*k]]))
    clean=next(i for i in range(1000) if all(separated_digits(v[i,j]) for j in display_indices))
    for predobj,num in [(a,'5'),(final,'6b')]:
        rows=[]; labels=[]
        for k in [1,2,3]:
            xx,yy=window(v[clean],k=k); pp=predobj['preds'][str(k)][clean]
            rows.extend([torch.cat((xx,pp[:2])),torch.cat((xx,yy[:2]))]); labels.extend([f'{k}x forecast',f'{k}x truth'])
        grid(rows,labels,FIG/f'Figure{num}.png',f'Figure {num} — video {9000+clean}, seed 0',[f'input {j+1}' for j in range(5)]+['h=1','h=2'])
    # Save exact identifiers and pixel diagnostics for reproducibility.
    diag=dict(figure2=[9000+i for i in example],overlap_candidate=9000+example[-1],worst={g:[9000+i for i in ids[g]] for g in ids},speed_video=9000+clean,speed_true_frame_numbers=[j+1 for j in display_indices])
    i=ids['R'][0]; p=a['preds']['R'][i]; q=transform(a['preds']['1'][i],1)
    py=transform(y[i],1); f=final['preds']['R'][i]
    diag['q27']=dict(video=9000+i,transform='R',consistency=float((p-q).square().mean()),prediction_peak=p.amax((1,2,3)).tolist(),truth_peak=py.amax((1,2,3)).tolist(),prediction_mass=p.sum((1,2,3)).tolist(),truth_mass=py.sum((1,2,3)).tolist(),prediction_error=float((p-py).square().mean()),final_error=float((f-py).square().mean()),final_consistency=float((f-transform(final['preds']['1'][i],1)).square().mean()))
    save(ART/'figure_selection.json',diag)
    river_diagram()
    return diag

def save(p,obj): Path(p).write_text(json.dumps(obj,indent=2))

def river_diagram():
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(11,7)); ax.set(xlim=(0,11),ylim=(0,7)); ax.axis('off')
    def box(x,y,w,h,text,color='#e3eff8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#31516d')); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.4,color='#31516d'))
    box(.2,5.6,2.4,1,'S1–S4, T1 levels\nB x 72 x 5 x 3\n[value, mask, age]')
    box(3.0,5.6,2.4,1,'R1, R2 rainfall\nB x 72 x 2 x 3\n[value, mask, age]')
    box(5.8,5.6,2.4,1,'Past actual releases\nB x 72 x 1 x 3\n[value, mask, age]')
    box(8.6,5.6,2.1,1,'Announced schedule\nB x 6 x 1 x 3\n[value, known, lead]')
    box(.8,3.8,7.1,1,'As-of joins; 10-minute bins; training-only scaling\nNode projection W: sensor/catchment/source features -> B x 72 x 6 x 64')
    for a in [(1.4,5.6),(4.2,5.6),(7,5.6)]: arrow(a,(4.3,4.8))
    box(8.6,3.8,2.1,1,'6-step schedule GRU\nB x 64'); arrow((9.65,5.6),(9.65,4.8))
    box(.8,2.05,7.1,1,'3 directed lag-aware graph blocks + causal node GRU\n72 lags; flow + ancestor + self edges (E=17); B x 6 x 64')
    arrow((4.3,3.8),(4.3,3.05))
    box(8.6,2.05,2.1,1,'Static geography\n6 x 2 coordinates\n17 edges + distances')
    arrow((8.6,2.55),(7.9,2.55))
    box(2.9,.65,5.4,1,'Select S1–S4; concatenate schedule state\nB x 4 x 128 -> shared + station forecast heads\nresidual to last level -> B x 4 (level at t + 60 min)',color='#dcf0e1')
    arrow((4.3,2.05),(5.4,1.65))
    ax.plot([10.7,10.95,10.95],[4.3,4.3,1.15],color='#31516d',lw=1.4)
    arrow((10.95,1.15),(8.3,1.15))
    ax.text(.3,.06,'Nodes: reservoir, S1, S2, S3, S4, T1. Main river: reservoir -> S1 -> S2 -> S3 -> S4.\nT1 joins before S2; unknown tributary joins before S3. Ancestor edges preserve downstream direction.',fontsize=7)
    fig.savefig(FIG/'Q1_architecture.png'); plt.close(fig)

class Document:
    def __init__(self):
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        self.styles=getSampleStyleSheet(); self.styles['BodyText'].fontName='Helvetica'; self.styles['BodyText'].fontSize=9; self.styles['BodyText'].leading=13
        self.styles['BodyText'].spaceAfter=7
        self.story=[]; self.md=[]
    def h(self,s,level=1):
        from reportlab.platypus import Paragraph,Spacer
        self.story.extend([Spacer(1,8),Paragraph(html.escape(s),self.styles['Heading'+str(min(level,3))])]); self.md.append('#'*level+' '+s+'\n')
    def p(self,s):
        from reportlab.platypus import Paragraph
        self.story.append(Paragraph(html.escape(s).replace('\n','<br/>'),self.styles['BodyText'])); self.md.append(s+'\n')
    def page(self):
        from reportlab.platypus import PageBreak
        self.story.append(PageBreak())
    def table(self,title,headers,rows,widths=None):
        from reportlab.platypus import Paragraph,Table,TableStyle,Spacer
        from reportlab.lib import colors
        self.h(title,2)
        cells=[[Paragraph(html.escape(str(s)),self.styles['BodyText']) for s in row] for row in [headers]+rows]
        table=Table(cells,colWidths=widths or [493/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dceaf4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#31516d')),('LINEBELOW',(0,1),(-1,-1),.2,colors.lightgrey),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
        self.story.extend([table,Spacer(1,8)])
        self.md.extend(['| '+' | '.join(map(str,headers))+' |','|'+'---|'*len(headers)]+['| '+' | '.join(map(str,r))+' |' for r in rows]+[''])
    def fig(self,name,caption):
        from reportlab.platypus import Image,KeepTogether,Paragraph,Spacer
        path=FIG/name
        im=PILImage.open(path); w,h=im.size; scale=min(493/w,560/h)
        self.story.append(KeepTogether([Image(str(path),width=w*scale,height=h*scale),Paragraph(html.escape(caption),self.styles['BodyText']),Spacer(1,9)]))
        self.md.append(f'![{caption}](figures/{name})\n')
    def finish(self):
        from reportlab.platypus import SimpleDocTemplate
        from reportlab.lib.pagesizes import A4
        def footer(canvas,doc):
            canvas.setFont('Helvetica',8); canvas.drawString(42,25,'AI651 Assignment 2 | Reproducible two-seed experiment'); canvas.drawRightString(A4[0]-42,25,str(doc.page))
        doc=SimpleDocTemplate(str(REPORT/'Final_Report.pdf'),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=42,title='Motion or Memory? River Forecasting and ConvLSTM Stress Tests',author='')
        doc.build(self.story,onFirstPage=footer,onLaterPages=footer)
        (REPORT/'Final_Report.md').write_text('\n'.join(self.md))

def q1(d):
    d.h('Question 1 — River Forecasting Proposal')
    d.h('1. Architecture diagram',2)
    d.fig('Q1_architecture.png','Diagram Q1. Every historical input is joined as of forecast issuance. Node and feature axes are explicit; gauges are projected to catchments without implying they are river stations. Future schedules use a separate encoder.')
    d.h('2. Assumptions',2)
    assumptions=[
    'A1. At least two complete hydrological years of ten-minute records are available, including some floods. This is an assumption about data volume, not a supplied fact. With less data, shrink the network and report wider uncertainty; flood performance cannot be promised.',
    'A2. Measurement time, arrival time and schedule announcement time can be recovered. A reading is usable only when it arrived by issuance. If historical arrival times are absent, impose a conservative fixed delay and state that backtest realism is limited.',
    'A3. Units, station datums and clock conventions can be reconciled. Standardise each continuous feature using training-period statistics; datum changes require explicit correction, since an apparent jump is not river dynamics.',
    'A4. A 12-hour history (72 bins) captures the main dependencies relevant to a one-hour forecast. This is a tunable operational assumption, validated chronologically; extend it if slow propagation exceeds 12 hours.',
    'A5. Dominant transport follows the supplied downstream river topology, with no strong reverse backwater effect over this horizon. Thus upstream-to-downstream edges are justified. If tides/backwater matter, add documented reverse edges and boundary levels; the present graph would otherwise be misspecified.',
    'A6. Travel times vary but relevant delays can be represented within the 72-bin lag bank. Do not equate station index with distance or impose a single unknown travel speed. Distances and recent levels condition learned lag weights.',
    'A7. Rain gauges provide imperfect catchment proxies; they do not measure future rainfall. Soft catchment weights use gauge locations and learned corrections. Unmonitored inflow remains partly unobservable, especially for S3 and S4.',
    'A8. Future schedules available at issuance have predictive value but are not guarantees of actual releases. A known/unknown flag and lead-time features distinguish genuine zero releases from absent schedules.',
    'A9. Reporting latency is usually below one ten-minute interval and short outages occur. Masks and ages are mandatory; severe station outages activate a conservative persistence fallback and an operator flag. A continuously staffed flood operation and a historical alert threshold register are assumed for deployment.'
    ]
    for s in assumptions: d.p(s)
    d.p('Supplied facts: levels at five stations, two located rain gauges, past releases and pre-announced schedules, station locations/connectivity/distances, ten-minute issuance, and one-hour predictions at S1–S4. No dataset, precise travel times, unmonitored tributary inflow or future rainfall is supplied.')
    d.h('3. Forecasting task and data specification',2)
    d.p('Let t be the wall-clock issuance on the ten-minute grid. Use bins ending t-710,...,t minutes (H=72); missing latest values remain missing. Predict the instantaneous/station-reported level y_i(t+60), i in {S1,S2,S3,S4}, not an average over the next hour. B is the batch size; historical lag, node, channel and future-lead axes are distinct.')
    d.table('Input/output specification',['Quantity','Shape / axes','Time and availability'],[
    ['Water levels','B x 72 x 5 x 3; [value,mask,age]','S1–S4,T1; measurement <= t and arrival <= t'],
    ['Rainfall','B x 72 x 2 x 3; [depth,mask,age]','R1,R2; causal ten-minute accumulation; arrival <= t'],
    ['Actual reservoir release','B x 72 x 1 x 3','Past effective bins, recorded/arrived by t'],
    ['Future release schedule','B x 6 x 1 x 3; [release,known,lead]','Effective t+10,...,t+60; latest announcement <= t per effective bin'],
    ['Static graph/geography','6 nodes, E=17 directed/self edges; node coordinates and catchment links','4 main-river edges, T1->S2, 6 self edges, 6 main-river ancestor shortcuts'],
    ['Node embeddings','B x 72 x 6 x 64','Causal projection of aligned inputs and static features'],
    ['Target / output','B x 4, station axis S1–S4','Target level at t+60; observed later, never an input']
    ],[112,174,207])
    d.p('Use a canonical time zone and event-time bins. Aggregate rain depth by summation, release by interval mean and level by the latest valid reading within a bin; never interpolate from a later reading. At every historical issuance rebuild the as-of view, including superseded schedule versions. A gap may carry forward the last value only with mask=0 and increasing age. Fit normalisation and imputation constants on training data only. Split chronologically, keep events intact, and purge at least history plus horizon at split boundaries.')
    d.h('4. Model computations and design justification',2)
    d.p('Let v denote one of six graph nodes, i a forecast station, s an observed bin, r a lag and a_uv the fixed downstream/self adjacency. z_v,s = W_v [aligned levels, gauge-weighted rain, actual release, masks, ages, static features] + b_v in R^64. Zero-fill missing normalised values only after adding mask and age; nonlocal station levels are not silently copied into every node. Rainfall at a node is sum_j w_vj rain_j, with nonnegative catchment weights informed by locations and learned from historical data.')
    d.p('Lag-aware block: e_uvrs = phi(z_v,s, z_u,s-r, river_distance_uv, r); alpha_uvrs = softmax over allowed upstream u and causal lags r in 0..71; m_v,s = sum_(u,r) alpha_uvrs W_msg z_u,s-r. Invalid or unavailable lag positions are masked. h_v,s = GRU([z_v,s,m_v,s], h_v,s-1). Repeat three graph/temporal blocks with residual projections. The fixed 17-edge graph contains five direct flow edges, six self edges and six main-river ancestor shortcuts (reservoir->S2,S3,S4; S1->S3,S4; S2->S4). Ancestor edges encode hydrological reachability, not instantaneous transport; learned lags still determine timing. They prevent a four-edge reservoir-to-S4 path from exceeding the three-block spatial reach. T1 can reach S4 through its three direct downstream edges.')
    d.p('The schedule GRU encodes six issued future release entries into q_t in R^64; missing schedule bins retain a known flag. Select h_i,t for S1–S4 and concatenate q_t, producing B x 4 x 128. A station-conditioned head gives delta_i = MLP_i([h_i,t,q_t]); yhat_i(t+60) = last_valid_level_i(t) + delta_i. If there is no valid level in the history, use a training-period datum fallback and flag it. Shared features improve data efficiency, while station heads allow different rating/response relationships. The delta skip reduces the burden of learning near-persistence.')
    d.p('Train with L = (1/(4B)) sum_(b,i) |yhat_bi-y_bi| on available labels, normalising per video/issuance by the number of valid stations. If all four labels are present this is ordinary MAE; batches without a valid station label are excluded. Report level MAE/RMSE by station, lead (here fixed at one hour), flood/nonflood strata and event. Use chronological validation to tune width/history and compare against last level and a linear autoregression. Do not infer hydraulic conservation from a learned graph: graph direction/catchment assignment are physical assumptions, but delay weights, nonlinear stage responses and rain-to-inflow conversion are empirical mappings.')
    d.p('Upstream levels propagate through directed lag paths; T1 contributes downstream of its junction at S2, and then S3/S4. Rain can affect all corresponding catchments; the missing tributary is represented only through indirect level/rain evidence. Reservoir history captures realised forcing, while the separate schedule context provides planned forcing that a purely historical model would miss. No future rainfall or future measured level appears in any equation.')
    d.h('5. Operation and exceptional conditions',2)
    for s in [
    '(i) Missing observations: retain value/mask/age channels, train with realistic block outages, and avoid treating absent rain or release as zero. Forecast from other nodes when possible. If S1–S4 are all stale beyond a validated outage threshold, issue a flagged persistence forecast with expanded empirical error bands rather than an unqualified network value.',
    '(ii) Delayed observations: use arrival-time filtering; an observation stamped before t but arriving after t is unavailable for this forecast. Incorporate it at the next issuance and replay causal history if needed. Backtests must mimic this rule, including the same delayed-data policy.',
    '(iii) Announced future releases: select only schedule versions announced by t. Map their effective periods to the six future bins, using duration-weighted rates for partial bins and preserving unknown flags. A release announced after t is excluded even if effective within the forecast horizon.',
    '(iv) Scheduled versus actual releases: retain separate fields and never overwrite a historical schedule with the realised future release. At the next issuance update actual-release history; learn residual schedule error statistics by lead and operator regime. Simulate plausible deviations for stress scenarios, label them as scenarios and widen conditional residual bands. Schedule forecasts can fail when operations deviate abruptly.',
    'Every ten minutes, fetch the as-of snapshot, validate, update encoders, score four stations and log both input version IDs and forecasts. Six nodes, 72 bins and width 64 give a small bounded graph; even all-node causal lag mixing is at most O(B H^2 E d) and the chosen sparse graph reduces constants. This is plausibly feasible within ten minutes, but no latency benchmark is claimed. Require a deployment latency/load test, monitoring, an input deadline and fallback if ingestion or inference misses the deadline.'
    ]: d.p(s)
    d.h('6. Uncertainty, limitations, and additional information',2)
    d.table('Limitations and requested information',['Issue / request','Forecast impact and response','Usable without?'],[
    ['Unmonitored tributary; request a gauge or inflow estimate','Unseen surges can bias S3/S4 low. Use indirect rain/level evidence and flag wet events; irreducible uncertainty remains.','Yes, with explicit blind spot'],
    ['Changing travel times; request flow, channel cross-sections, roughness','Lag attention can adapt to level regime but may extrapolate poorly in extreme floods. Validate eventwise; distances alone do not identify delays.','Yes; lag bank remains empirical'],
    ['Rainfall coverage; request radar, catchment boundaries, forecast rain','Two gauges miss spatial storms; no available-at-issuance future rain means flood growth can be missed. Add forecasts only with archived issue times.','Yes, degraded storm lead'],
    ['Operational deviations; request schedule revisions and actual release logs','Announcement/effective/arrival histories prevent leakage and quantify schedule reliability; emergency changes remain unpredictable.','Yes; unknown schedules masked'],
    ['Quality and latency; request datum changes, maintenance, arrival logs','Spikes and stale readings distort levels and delay inference. QC, outage augmentation, masks and fallback help.','Conditional; reliable timestamps/units essential'],
    ['Rare extremes and regime shifts; request long records and event labels','Global MAE can hide peak failure. Event-stratified assessment and drift monitoring are required.','Yes for monitored pilot; extreme-flood claims unvalidated'],
    ['Backwater/tides; request downstream controls, boundary levels','Reverse effects violate A5; add edges and forcings if documented.','Conditional on A5 validity'],
    ['Uncertainty calibration; request operational alert thresholds','Use rolling held-out signed residual quantiles per station/regime around point forecast for empirical 80/95% intervals. Coverage under shift is not guaranteed; validate peaks and recalibrate.','Yes; report levels and bands, avoid uncalibrated probabilities']
    ],[120,285,88])
    d.p('This is a design proposal, as requested: no river data, river training result, guaranteed flood probability or measured deployment latency is claimed. Empirical residual intervals describe observed errors and do not resolve unseen tributary forcing or guarantee coverage during unprecedented events.')

def build(reuse_figures=False):
    from torchvision.datasets import MovingMNIST
    REPORT.mkdir(exist_ok=True)
    test=MovingMNIST(ROOT/'data',split=None,download=False).data[9000:10000]
    ids=json.loads((ART/'figure_selection.json').read_text()) if reuse_figures else make_figures(test)
    flow=json.loads((ART/'flow_summary.json').read_text()); hw=json.loads((ART/'hardware.json').read_text())
    d=Document(); d.h('Motion or Memory?'); d.h('AI651 — Deep Learning for Space, Time and Graphs',2)
    d.p('Assignment 2 | River forecasting proposal and Moving MNIST experiments | Prepared for Zain | 29 October 2026 (Asia/Karachi).')
    d.p('Code repository: https://github.com/ZainGulbaz/DLST-PA2. The project includes the experiment source, final model weights, results, figures and prediction-commit history. LMS submission has not been performed.')
    d.p('All required models A–D were trained for 30 epochs with seeds 0 and 1. The final model is symmetrised D, with no additional training. Every model metric below uses the 1,000 held-out videos, except Table 1 synthetic checks, Figure 1 training-only optical flow, parameter counts and timings. Tables use mean (seed 0, seed 1); deterministic references use a single value. Errors are rounded to three significant figures, with scientific notation below 1e-3. Underlying unrounded results are included in artifacts/.')
    d.p('The report includes six numbered experimental tables, six numbered experimental figures (some multi-panel), all Q2.1–Q2.17 answers, a causal forecasting proposal and GIFs. Pre-experiment predictions were committed before the corresponding experiments; their exact text and commit history are retained.')
    d.page(); q1(d); d.page()
    d.h('Question 2 — Motion or Memory?')
    d.h('Protocol and implementation',2)
    d.p(f"Hardware: {hw['gpu']}; CUDA {hw['cuda']}; PyTorch {hw['torch']}; CUDA_VISIBLE_DEVICES={hw['visible_devices']}. Float32 training/inference, TF32 disabled, four CPU threads, cuDNN autotuning enabled. Bitwise cross-platform determinism is not claimed. Seeds control weights, video shuffling, windows and augmentation. Training uses Adam lr=0.001, batch size 32, 30 complete shuffled passes over 8,000 videos (250 batches/epoch), no schedule, final-epoch weights. Loss averages valid pixel/frame errors within each video, then videos equally. No teacher forcing, AMP, test-based selection or early stopping is used.")
    d.p('Dataset: torchvision MovingMNIST(split=None), official file with MD5 be083ec986bfe91a449d63653c411eb2; only transport was upgraded to HTTPS. Video indices 0–7999 train, 8000–8999 validation and 9000–9999 test. D was fixed analytically before tests; no candidate selection used test or validation. Float32 frames are divided by 255. Training starts: A/B uniform 1–11 at 1x; C/D speeds uniform {1,2,3}, starts uniform 1–14,1–8,1–2 respectively. B samples one uniform square symmetry per window. Extra predictions always exist but unavailable targets are masked. Every cross-speed comparison uses E(2).')
    d.p('The exact baseline uses pixel-unshuffle(4), two biased 3x3 ConvLSTM cells of width 64, a biased 1x1 64->16 head, pixel-shuffle(4) and sigmoid. The gates follow the supplied no-peephole equations. Nine recurrent steps consume five observations and four own predictions; only step 5–9 outputs are scored. Gradients flow through feedback. Model D changes layer 2 dilation to 2 with padding 2; layer 1 remains dilation 1. No other architecture or protocol changes are made.')
    checks=json.loads((ART/'table1_checks.json').read_text()); rows=[]
    expected=['0; equal=True','0; equal=True','0; equal=True','(7,10,43)','(7,43,10)','1–5 / 6–10','1,3,5,7,9 / 11,13,15,17,19','1,4,7,10,13 / 16,19']
    for r,exp in zip(checks,expected):
        actual=f"{r['max_abs']:g}; equal={r['equal']}" if 'max_abs' in r else str(r['position']) if 'position' in r else f"{r['inputs']} / {r['targets']}"
        rows.append([r['check'],exp,actual])
    d.table('Table 1 — Exact transformation and window checks',['Check','Expected','Measured'],rows,[40,200,253])
    d.fig('Figure1.png',f"Figure 1. Prescribed Farneback flow from the first 50 training videos, all 950 consecutive pairs; {flow['pixels']:,} kept pixels; median speed {fmt(flow['median_speed'])} pixels/frame. Both histograms report fractions of kept pixels. Mask: both uint8 intensities >51 and magnitude >0.5. Settings: (0.5,3,15,3,5,1.2,0).")
    fractions=np.array(flow['direction_fractions'])
    variation=float(np.abs(fractions-1/18).sum()/2)
    direction=np.load(ART/'flow.npz')['direction']
    edges=np.arange(0,361,20)
    mirror_tv=float(np.abs(fractions-np.histogram((180-direction)%360,bins=edges)[0]/len(direction)).sum()/2)
    rotation_tv=float(np.abs(fractions-np.histogram((direction+90)%360,bins=edges)[0]/len(direction)).sum()/2)
    d.h('Q2.1 — Familiarity of motion',2)
    d.p(f"Figure 1(a) spreads probability across the circle; the total variation from a uniform 18-bin distribution is {fmt(variation)}. Mirroring maps theta to (180-theta) modulo 360; anticlockwise rotation maps theta to theta+90. A broadly uniform directional distribution makes both sets of trajectories broadly familiar, even though digit appearance is different. Figure 1(b)'s typical speed is {fmt(flow['median_speed'])}; 2x and 3x correspond roughly to {fmt(2*flow['median_speed'])} and {fmt(3*flow['median_speed'])} pixels per observed frame. These multiples describe unreflected constant-velocity segments; bounces and optical-flow estimation can reduce net displacement. Overlap, the two-frame brightness mask and the >0.5 threshold bias this pixel-weighted estimate; it is not a ground-truth per-digit velocity distribution. Without an overlap-free sensitivity analysis we cannot rule out a change in detailed directional weights, though broad directional coverage is visible.")
    d.p(f"The histogram is not invariant: recomputing the same 20-degree-bin histogram after mapping the measured angles gives total variation {fmt(mirror_tv)} under mirror and {fmt(rotation_tv)} under rotation relative to the original. Thus mirrored and rotated directions are represented, but their frequencies are not identical in this finite sample. Rotation in particular introduces a measurable motion-statistics shift as well as an appearance shift; this limits causal attribution to appearance alone.")
    d.h('Q2.2 — Controlled transformations',2)
    d.p('Mirror and rotation preserve physical motion validity and speed magnitude but transform direction, wall orientation and digit appearance. If the direction distribution is approximately isotropic, their major distribution shift is appearance/orientation, rather than an entirely new motion law. Speed-up preserves each sampled digit appearance but changes displacement and temporal aliasing, including possible unobserved bounces. Thus these interventions probe complementary vulnerabilities; they do not perfectly isolate appearance because the learned representation and boundary response can also be orientation-sensitive.')
    d.table('Table 2 — Original 1x baseline',['Metric','A: mean (s0,s1)','Blank','Copy-last'],[[f'MSE {h+1}',pair('A','mse_h',index=h),fmt(load('Blank')['mse_h'][h]),fmt(load('Copy-last')['mse_h'][h])] for h in range(5)]+[['E(5)',pair('A','E5'),fmt(load('Blank')['E5']),fmt(load('Copy-last')['E5'])],['Parameters','480,784','0','0']],[83,220,95,95])
    d.fig('Figure2.png',f"Figure 2. Seed-0 A, original 1x, videos {ids['figure2']}; upper row in each pair is five inputs then predictions, lower is inputs then truth. Video {ids['overlap_candidate']} was selected for merging/separating foreground components as an overlap example. All indices are global dataset indices.")
    d.h('Q2.3 — One-step reach of A',2)
    d.p('One folded-grid cell is 4 original pixels centre-to-centre. Each 3x3 gate convolution extends spatial radius by one cell. The input u_tau can reach the output through the layer-1 gate convolution and then the layer-2 gate convolution: radius 2 cells, or 8 original pixels per coordinate (Chebyshev radius). The same longest path exists from h1_(tau-1) through both convolutions; h2_(tau-1) traverses only the second, for radius 4 pixels. Cell-state updates are pointwise and do not add reach. The 1x1 head and shuffle operations add none. Recurrence accumulates reach over time, but those earlier steps are not counted inside this one update. At a diagonal the Euclidean centre distance can be 8*sqrt(2) pixels; the stated 8 is the axis/Chebyshev convention used to compare displacements.')
    d.h('Q2.4 — Baseline accuracy and overlap',2)
    d.p(f"Table 2 gives A E(5)={pair('A','E5')}, versus Blank {fmt(load('Blank')['E5'])} and Copy-last {fmt(load('Copy-last')['E5'])}. A's MSE changes from {pair('A','mse_h',index=0)} at h=1 to {pair('A','mse_h',index=4)} at h=5. Figure 2 makes the horizon-dependent localisation and sharpness error visible. Pixel MSE rewards an averaged, blurry forecast when digit position or overlap identity is uncertain; successful low MSE does not establish that individual digits are tracked through occlusion.")
    if ids['overlap_candidate']==9002:
        d.p('In Figure 2 video 9002, the true 1-like stroke meets the top of the 5-like digit near the last displayed future frames. A blurs away the lower digit structure and spreads intensity between the two objects; by h=5 the two-digit scene is represented as a diffuse near-vertical cloud. This specific overlap example does not show successful recovery of two digit identities after occlusion.')
    d.h('Q2.5 — Prediction recorded before baseline transformation/speed tests',2)
    d.p((ROOT/'predictions/Q2_5.md').read_text().strip())
    d.p('The paragraph above is copied without editing from predictions/Q2_5.md; the experiment committed it before running the mirror/rotation/speed evaluation. See prediction_commits.txt for commit hashes and timestamps.')
    d.table('Table 3 — Baseline consistency and correctness',['Metric','A: mean (s0,s1)','Blank','Copy-last'],[[key,pair('A',key),fmt(load('Blank')[key]),fmt(load('Copy-last')[key])] for key in ['E5','CM','EM','CR','ER']],[83,220,95,95])
    for g in ['M','R']:
        for rank in [1,2]: d.fig(f'Figure3_{g}_{rank}.png',f"Figure 3 ({g}, rank {rank}). Seed-0 A, one of the two largest per-video C_{g} values among all 1,000 test videos. Rows and ten columns follow the requested format. GIF: report/gifs/Figure3_{g}_{rank}.gif; five input columns stay fixed while the forecast column steps through h=1..5.")
    d.fig('Figure4.png','Figure 4. Seed-0 A per-video mirror and rotation consistency-gap histograms across all 1,000 test videos. Vertical lines mark exactly the videos in Figure 3; the matching GIFs are included in report/gifs/.')
    d.h('Q2.6 — Consistency alone is insufficient',2)
    d.p('Blank always produces zero, and spatial transformation of zero remains zero. Copy-last simply selects and repeats the last frame, which commutes exactly with a spatial permutation. Their CM=CR=0 in Table 3 is therefore compatible with poor forecasts. C_T measures agreement between two predictions, not agreement with reality; E_T must accompany it. A consistently stationary or blurry forecast is not evidence of learned motion.')
    diag=ids['q27']
    d.h('Q2.7 — A specific disagreement',2)
    d.p(f"Consider dataset video {diag['video']}, Figure 3(R, rank 1). Rows 3 and 4 compare predicting after rotation against rotating the original prediction: their per-video squared disagreement is {fmt(diag['consistency'])}. For the direct rotated-input prediction, peak intensity changes from {fmt(diag['prediction_peak'][0])} at h=1 to {fmt(diag['prediction_peak'][-1])} at h=5, versus true-frame peaks {fmt(diag['truth_peak'][0])} and {fmt(diag['truth_peak'][-1])}. Its foreground intensity mass changes from {fmt(diag['prediction_mass'][0])} to {fmt(diag['prediction_mass'][-1])}, versus truth {fmt(diag['truth_mass'][0])} to {fmt(diag['truth_mass'][-1])}. The panel shows the corresponding spatial disagreement: the two routes fail to maintain the same sharply localised moving-digit support. These are representation/localisation failures; the figure alone does not establish a particular incorrect bounce law or object identity assignment.")
    if diag['video']==9622:
        d.p('Visual inspection of video 9622: in row 3 the rotated 9-like digit broadens vertically by h=4–5 and leaves an extra diffuse blob below it, while the short horizontal 1-like stroke splits and fades into two low-intensity pieces. In row 4 the upper digit instead spreads into a wide horizontal wedge and the lower stroke becomes a larger cloud toward the bottom-left. Both disagree with the two sharply separated true digits in row 2. Thus the transformation changes both predicted shape and localisation; neither route maintains a reliable two-object account.')
    d.h('Q2.8 — Rotation versus mirror',2)
    harder='rotation' if mean('A','ER')>mean('A','EM') else 'mirror'
    d.p(f"For A at 1x, transformed errors are EM={pair('A','EM')} and ER={pair('A','ER')}; {harder} is harder by the two-seed mean transformed error. The corresponding gaps CM={pair('A','CM')} and CR={pair('A','CR')} quantify consistency separately. Broad direction coverage in Figure 1 weakens a pure unfamiliar-direction explanation; altered digit orientation and the architecture's orientation-sensitive features are plausible drivers. This is evidence consistent with appearance sensitivity, not a causal proof that appearance alone explains the difference.")
    d.table('Table 4 — Baseline speed stress test',['Metric / speed','A: mean (s0,s1)','Copy-last'],[[key,pair('A',key),fmt(load('Copy-last')[key])] for key in ['E2_1','E2_2','E2_3','E5','E5_2']],[110,260,123])
    d.fig('Figure5.png',f"Figure 5. Seed-0 A, video {ids['speed_video']}; inputs and h=1,2 prediction/truth at all three speeds. The same video is reused in Figure 6(b). All displayed true frames have two foreground components above intensity 0.05 (area >12 pixels) whose bounding boxes are separated by at least three pixels; global frame numbers inspected: {ids['speed_true_frame_numbers']}.")
    d.h('Q2.9 — Speed failure and reach',2)
    d.p(f"A's E(2) is {pair('A','E2_1')} at 1x, {pair('A','E2_2')} at 2x, and {pair('A','E2_3')} at 3x (Table 4). Figure 5 shows how the same video changes under temporal subsampling. Larger displacement makes localisation and sharpness harder, but an 8-pixel structural reach is not a guaranteed speed threshold: velocities vary, diagonal versus axial distance matters, and previous recurrent updates can retain information. A visible blur suggests conditional averaging; stalling suggests copying/persistence; duplication suggests incompatible positional hypotheses; disappearance suggests loss of a digit representation. None alone proves a reach bottleneck. We therefore describe the measured deterioration rather than declaring failure at a universal speed.")
    if ids['speed_video']==9029:
        d.p('In Figure 5 video 9029, A loses the internal shape of the 8-like digit even at 1x. At 3x its two futures are two diffuse blobs clustered near the preceding scene, whereas the true 2-like digit moves far down and left and the 8-like digit is sharply separated above/right. The lower-left displacement is substantially under-predicted: this is a combination of stalling and blur, with no visible duplicate digit needed to explain this example. There is already a large measured deterioration at 2x, where typical displacement is below the nominal 8-pixel reach, showing that reach alone missed the training-distribution and representation problem.')
    d.h('Q2.10 — Separate unfamiliar speed from reach',2)
    d.p('A alone cannot distinguish the two. Train C on the uniform speed mixture using exactly A architecture and protocol, then compare to A by E(2) at each speed. Improvement implicates training distribution. Compare C with D on the same mixture and budget to test a larger-reach architecture. For stronger attribution, bin by independently measured nonoverlapping displacement relative to the respective reaches and use several matched-capacity receptive-field designs; dilation also changes spatial sampling and optimisation, so C versus D is not a perfect intervention on reach alone.')
    d.h('Q2.11 — Prediction recorded before B–D training',2)
    d.p((ROOT/'predictions/Q2_11.md').read_text().strip())
    d.p('The paragraph above is copied without editing from predictions/Q2_11.md, committed after the baseline tests and before B–D training. No model was retrained or chosen using test performance.')
    columns=['E5','CM','EM','CR','ER','E2_1','E2_2','E2_3','parameters','training_seconds_per_epoch','inference_seconds']
    labels=['E(5), 1x','C_M','E_M','C_R','E_R','E(2), 1x','E(2), 2x','E(2), 3x','Parameters','Train sec/epoch','Inference sec/1000']
    d.table('Table 5 — What data and reach change',['Metric']+list('ABCD'),[[label]+[pair(n,key) for n in 'ABCD'] for key,label in zip(columns,labels)],[91,100.5,100.5,100.5,100.5])
    d.p('Parameter counts are exact (480,784 for each model; the repeated seed formatting is retained for consistency). Training times are the wall-clock mean of all 30 epochs, excluding download; per-epoch checkpoint writing is outside the timer. Inference times include GPU synchronisation around every batch, after float32 batches reach the device, under eval/no_grad; one warm-up batch is excluded. At batch 32 the final batch contains 8 videos. Time variability is shown by seed, not interpreted as a causal accuracy effect.')
    d.h('Q2.12 — Model D and its larger reach',2)
    d.p('D retains patch size 4, 64 hidden channels and every baseline layer. Layer 1 uses 3x3 dilation 1 (radius 1 folded cell), while layer 2 uses 3x3 dilation 2 (radius 2 folded cells), with padding 2 so shapes are unchanged. The longest within-step path from u_tau or h1_(tau-1) through both gate convolutions has radius (1+2)*4=12 original pixels per coordinate, strictly above A/C at 8; the path from h2_(tau-1) has radius 8. Pointwise memory and the head add none. Combining layer-1 offsets {-1,0,1} with layer-2 offsets {-2,0,2} covers every integer offset -3..3 per coordinate, avoiding holes on the longest path. D has exactly 480,784 parameters, below the 721,176 cap. Its Euclidean diagonal maximum is 12*sqrt(2).')
    d.h('Q2.13 — Attribution supported by comparisons',2)
    d.p(f"Appearance/orientation: compare A with B, whose only intervention is uniform square-symmetry augmentation. EM changes from {fmt(mean('A','EM'))} to {fmt(mean('B','EM'))} and ER from {fmt(mean('A','ER'))} to {fmt(mean('B','ER'))}; CM/CR remain learned rather than guaranteed. These comparisons measure orientation-sensitive generalisation, with the caveat about directional statistics in Q2.8. Unfamiliar motion: compare A with C, E(2) at 3x {fmt(mean('A','E2_3'))} -> {fmt(mean('C','E2_3'))}; improvement shows that training exposure can help without changing reach. Architectural reach: compare C with D on the identical mixture, E(2) at 3x {fmt(mean('C','E2_3'))} -> {fmt(mean('D','E2_3'))}. Only if D clearly improves is this evidence for a benefit of the larger-reach architecture; if it does not, the experiment does not justify labelling the residual error a demonstrated reach bottleneck. Two seeds describe variability and do not establish statistical significance.")
    d.p('If C remains poor at 3x while D is strong, that supports a limitation of the original spatial update, subject to the changed dilation/optimisation caveat. More data can change weights but cannot make a one-step computation depend on an input outside its structural support. It can still improve use of earlier recurrent information, so “more data cannot fix anything” would be too strong. Residual bounce, overlap and long-rollout errors may have mixed or unresolved causes rather than one exclusive label.')
    d.h('Q2.14 — Cost on ordinary clips',2)
    changes=', '.join(f"{n}: {fmt(mean(n,'E5'))} ({100*(mean(n,'E5')/mean('A','E5')-1):+.3g}% vs A)" for n in 'BCD')
    d.p(f"Original 1x E(5), Table 5: {changes}. A positive percentage is an accuracy cost. The tradeoff is worthwhile only if the deployment values orientation robustness and faster motion enough to accept that cost; this benchmark prioritises both, so the final model follows the specified D design rather than selecting a winner from the test set.")
    d.h('Built-in symmetry: short proof',2)
    d.p('Let rho(g) act by spatial pixel permutation and define fbar(X)=(1/8) sum_g rho(g)^(-1) f(rho(g)X). For any a in G, set h=ga; then g=h a^(-1) and g^(-1)=a h^(-1). Hence fbar(aX)=(1/8) sum_h a h^(-1) f(hX)=a fbar(X). Mirror and rotation are elements of G, so both consistency gaps are exactly zero in real arithmetic for any underlying f. The proof needs the complete group and correct inverses; for g=MR^a the inverse is itself, while (R^a)^(-1)=R^(-a). Floating-point addition in different orders can leave tiny nonzero squared gaps.')
    d.table('Table 6 — Symmetrised models (final = symD)',['Metric','symB: mean (s0,s1)','symD / final: mean (s0,s1)'],[[label]+[pair(n,key,scientific=key in ('CM','CR')) for n in ['symB','symD']] for key,label in zip(columns,labels)],[117,188,188])
    d.p('The final model is symmetrised D as specified in the handout. Optional D+B was not trained. The training-time entries copy the corresponding B or D time exactly; symmetrisation requires no additional training. Wrapped inference includes all eight input transforms, forward passes, inverse transforms and averaging; gaps are reported in scientific notation. Original and transformed correctness errors are always shown next to consistency.')
    for g in ['M','R']:
        for rank in [1,2]: d.fig(f'Figure6a_{g}_{rank}.png',f"Figure 6(a), {g} rank {rank}. Final symD, exactly the same test video as Figure 3 ({g}, rank {rank}). GIF: report/gifs/Figure6a_{g}_{rank}.gif. Rows/columns are the same as Figure 3.")
    d.fig('Figure6b.png',f"Figure 6(b). Final symD on the exact Figure 5 video {ids['speed_video']}, with the same windows and first two horizons.")
    d.h('Q2.15 — Learned versus built-in symmetry',2)
    d.p(f"B learns from augmented examples but its gaps are CM={pair('B','CM')} and CR={pair('B','CR')}. SymB reduces them to {pair('symB','CM',True)} and {pair('symB','CR',True)} by algebraic construction, independently of training success. Correctness still needs evaluation: B E(5)={pair('B','E5')}, symB={pair('symB','E5')}; D={pair('D','E5')}, symD={pair('symD','E5')}. At 1x the wrapped transformed errors match original errors to numerical precision because the transforms preserve pixel MSE. Inference for D is {pair('D','inference_seconds')} seconds versus symD {pair('symD','inference_seconds')}; it runs eight transformed predictions. Averaging can blur mutually displaced or differently shaped predictions, and can worsen an original orientation where the unwrapped model was especially accurate. Here the observed original E(5) change from D to symD is {100*(mean('symD','E5')/mean('D','E5')-1):+.3g}%; the sign states whether that harm occurred on the full test set, rather than assuming averaging always helps.")
    d.h('Q2.16 — Revisit the disagreement video',2)
    d.p(f"On the Q2.7 video {diag['video']}, the final rotation consistency gap is {fmt(diag['final_consistency'],True)} instead of A's {fmt(diag['consistency'])}. Its rotated-truth MSE changes from {fmt(diag['prediction_error'])} to {fmt(diag['final_error'])}. Figure 6(a) makes the remaining position/shape errors visible. Symmetrisation fixes agreement between the two transformation routes exactly; agreement does not mean the digit path is correct. Speed exposure and the larger-reach architecture are assessed globally through A/C/D in Table 5; this one final panel cannot uniquely attribute each visible improvement to one of those changes.")
    if diag['video']==9622:
        d.p('No, the final model does not get this example fully right. The 9-like digit remains a diffuse blob rather than a recognisable moving digit, and the short 1-like stroke fades markedly at later horizons. Rows 3 and 4 now agree, but they agree on this incorrect blurred forecast. Figure 6(b) video 9029 similarly retains substantial blur and merges intensity between the two digit locations at 3x. These examples limit any claim that the final model has learned reliable object-level motion.')
    d.h('Q2.17 — Final verdict (under 150 words)',2)
    d.p('The final model learns some predictive temporal structure, but does not establish a reliable object-level motion law. Its error improves on Copy-last, while the inspected examples retain severe blur and fading. Exact mirror/rotation consistency comes from group averaging, so it does not prove that D learned the motion rule. A/B and A/C test orientation and speed exposure; C/D tests a larger-reach update under matched data and parameter count. A skeptic should require correct transformed futures, robustness across individual videos, and successful separated-digit, bounce and overlap cases, not a small consistency gap alone. This experiment has not tested new digit families, longer rollouts or unseen acceleration. Tracking accuracy through those shifts would provide stronger evidence of transferable motion learning.')
    d.page(); d.h('Reproducibility and limitations')
    d.p('Run from the project root with CUDA_VISIBLE_DEVICES=2 MPLBACKEND=Agg ../../.venv/bin/python experiment.py. The pipeline validates transforms, downloads/checks the official data, trains and checkpoints every epoch, commits the required predictions before tests, evaluates both seeds and generates this report. A file lock prevents duplicate runs. Rerunning resumes final-epoch checkpoints and preserves recorded predictions; move artifacts/checkpoints aside to force a fresh experiment. The validation split was reserved and was not needed to fix the analytic D architecture.')
    commits=subprocess.check_output(['git','log','--format=%H %aI %s','--','predictions'],cwd=ROOT,text=True)
    (REPORT/'prediction_commits.txt').write_text(commits)
    d.p('Prediction commit records:\n'+commits.strip())
    d.p('Limitations: two seeds are not a confidence interval; no significance claim is made. Foreground-component screening identifies candidate overlap/clean examples but does not supply ground-truth digit identities or exact velocities. Optical flow may be biased by overlap and masking. No per-object tracking metric, out-of-family digit test, longer-horizon test or river-data benchmark was run. Code repository: https://github.com/ZainGulbaz/DLST-PA2; LMS upload has not been performed. Training wall-clock time includes batch construction and GPU execution, and excludes checkpoint serialisation; it can vary with host contention.')
    d.h('References',2)
    d.p('Assignment source: supplied AI651 Fall 2026 Assignment 2 PDF, DL4STG_PA2.pdf. ConvLSTM: Shi et al. (2015), Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting, NeurIPS. Moving MNIST: Srivastava et al. (2015), Unsupervised Learning of Video Representations using LSTMs, ICML. Dataset API and model operations were inspected from the installed torchvision/PyTorch implementation; no external numerical results were imported.')
    d.finish()
    save(REPORT/'completion_manifest.json',dict(models=['A','B','C','D','symB','symD'],seeds=[0,1],epochs=30,figure_selection=ids,tables=list(range(1,7)),figures=list(range(1,7)),report='Final_Report.pdf',public_repository='https://github.com/ZainGulbaz/DLST-PA2'))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-figures',action='store_true',help='Reuse existing figures and example selections for text-only report edits.')
    build(reuse_figures=parser.parse_args().reuse_figures)
