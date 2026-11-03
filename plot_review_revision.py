"""Reviewer-driven figures from saved results. No network or model calls.

Usage: MPLCONFIGDIR=/private/tmp/pde-review-mpl python3 plot_review_revision.py
       ... --archive /path/to/pde-runs-v1.zip
       ... --runs-dir data/recorded_runs
Raw-run figures accept the pinned ZIP or a locally manifested run directory.
"""
from pathlib import Path
import argparse
import hashlib
import gzip
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
IMG = ROOT / 'images'
MODELS = ['gemini-3.1-pro-preview', 'claude-opus-4-6', 'gpt-5.4',
          'gemini-3-flash-preview', 'claude-sonnet-4-20250514', 'gpt-5.4-mini']
LABELS = ['Gemini 3.1 Pro', 'Claude Opus 4.6', 'GPT-5.4',
          'Gemini 3 Flash', 'Claude Sonnet 4', 'GPT-5.4 mini']
SHORT = ['Gemini\n3.1 Pro', 'Claude\nOpus 4.6', 'GPT-5.4',
         'Gemini\n3 Flash', 'Claude\nSonnet 4', 'GPT-5.4\nmini']
DIFFS = ['easy', 'medium', 'hard']
COLORS = ['#226c9c', '#bd641d', '#24856b']
MCOLORS = ['#226c9c', '#bd641d', '#24856b', '#7953a1', '#a74553', '#6d6d6d']
summary = json.loads((ROOT/'paper_artifacts/summary.json').read_text())
stats = json.loads((ROOT/'paper_artifacts/statistical_analysis.json').read_text())
CELLS = {(c['model'], c['difficulty']): c for c in summary['cells']}
STATS = {(c['model'], c['difficulty']): c for c in stats['cells']}

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8.5,
    'axes.labelsize':8.5, 'axes.titlesize':9, 'xtick.labelsize':8,
    'ytick.labelsize':8.5, 'legend.fontsize':8, 'pdf.fonttype':42,
    'axes.spines.top':False, 'axes.spines.right':False})

def save(fig, name):
    fig.savefig(IMG/f'{name}.pdf', metadata={'Author':'Joseph K. Miller',
        'CreationDate':None, 'ModDate':None})
    fig.savefig(IMG/f'{name}.png', dpi=180)
    plt.close(fig)
    print(name)

def clean(ax, axis='x'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#e4e7e9', lw=.55)
    ax.tick_params(length=2.5)

def task_figure():
    fig, ax = plt.subplots(figsize=(6.75,3.05))
    fig.subplots_adjust(0,0,1,1)
    ax.set(xlim=(0,1), ylim=(0,1)); ax.axis('off')
    def box(x,y,w,h,title,body,fc,edge):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=.012',
                     fc=fc,ec=edge,lw=.9))
        ax.text(x+w/2,y+h-.045,title,ha='center',va='top',weight='bold',fontsize=9)
        ax.text(x+w/2,y+h-.11,body,ha='center',va='top',fontsize=8.5,linespacing=1.45)
    def arrow(a,b,label=None,dy=.025):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,
                                    lw=1.0,color='#485866'))
        if label:
            ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+dy,label,ha='center',va='bottom',fontsize=8)
    box(.02,.72,.29,.24,'Hidden problem',r'Coefficients $a,b,c,f$'+'\nSeed fixes the equation', '#f0f1f3','#6d7680')
    box(.70,.72,.28,.24,'Private evaluation',r'Compare $\hat\theta$ with true $\theta$'+'\nCoefficient error $E$', '#e8f2ed','#3c8065')
    arrow((.32,.87),(.69,.87),'reference coefficients',dy=.026)
    box(.02,.35,.27,.28,'Language model','Choose test functions\nRequest tools; inspect results\nContinue or submit', '#e9f2f9','#226c9c')
    box(.38,.35,.27,.28,'Numerical tools',r'Evaluate $u,u^{\prime}$; measure $\int f\varphi$'+'\nFit coefficients numerically\nReturn fits and diagnostics', '#fff3df','#ba8537')
    box(.74,.35,.24,.28,'Submission','Latest fitted\ncoefficient vector\n'+r'$\hat\theta$', '#e8f2ed','#3c8065')
    arrow((.30,.55),(.37,.55),'actions',dy=.025)
    arrow((.37,.40),(.30,.40))
    ax.text(.335,.30,'feedback',ha='center',fontsize=8)
    arrow((.66,.49),(.73,.49),'fit',dy=.025)
    arrow((.86,.64),(.86,.71))
    # Route from the hidden problem's right edge to the tools' top edge,
    # leaving the inter-row gutter and label clear of all box borders.
    from matplotlib.path import Path as ArrowPath
    route = ArrowPath([(.319,.75),(.515,.75),(.515,.640)],
                      [ArrowPath.MOVETO,ArrowPath.LINETO,ArrowPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=route,arrowstyle='-|>',
                mutation_scale=10,lw=1,color='#485866'))
    ax.text(.51,.805,'generated response\nand tool data',ha='center',
            va='center',fontsize=8,linespacing=1.15)
    ax.text(.5,.17,'One accepted stored query = one solver row = one unit of query budget',
            ha='center',weight='semibold',fontsize=8.5)
    ax.text(.5,.075,'Solution evaluations, term diagnostics, and checks are free within the turn limit.\n'
            'A check reveals a source moment but does not add a stored solver row.',
            ha='center',va='center',fontsize=8.3,linespacing=1.4)
    save(fig,'review_task')

def success_figure():
    fig, axs=plt.subplots(1,3,figsize=(6.75,2.65),sharey=True)
    fig.subplots_adjust(left=.19,right=.975,bottom=.20,top=.81,wspace=.22)
    for j,(ax,d) in enumerate(zip(axs,DIFFS)):
        for i,m in enumerate(MODELS):
            q=STATS[m,d]['success']['0.01']; x=q['rate']; lo,hi=q['wilson_95_interval']
            ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',ms=4,color=COLORS[j],
                        capsize=2,lw=1)
            ax.text(1.18,i,f"{q['count']}/10",ha='center',va='center',fontsize=8)
        ax.set(xlim=(-.03,1.38),ylim=(5.55,-.55),xticks=[0,.5,1],yticks=range(6))
        ax.set_title(d.capitalize(),weight='bold')
        ax.set_xlabel('Success fraction')
        if j==0: ax.set_yticklabels(LABELS)
        clean(ax)
    fig.text(.19,.97,'Final recovery: $E<0.01$  |  10 shared tasks per setting',fontsize=9,weight='semibold',va='top')
    save(fig,'review_success')

def confidence_figure():
    fig,axs=plt.subplots(2,3,figsize=(6.75,4.2),sharey=True)
    fig.subplots_adjust(left=.19,right=.975,bottom=.19,top=.83,wspace=.18,hspace=.56)
    for j,d in enumerate(DIFFS):
        for row,key in enumerate(['brier_score','signed_calibration_in_the_large_gap']):
            ax=axs[row,j]
            for i,m in enumerate(MODELS):
                item=STATS[m,d]['confidence']['run_balanced'][key]
                x=item['estimate'];lo,hi=item['whole_run_bootstrap_95_interval']
                if row==1:x,lo,hi=100*x,100*lo,100*hi
                ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',ms=3.8,capsize=2,lw=.9,color=COLORS[j])
            ax.set(yticks=range(6),ylim=(5.55,-.55))
            if j==0:ax.set_yticklabels(LABELS)
            if row==0:
                ax.set(xlim=(0,.6),xticks=[0,.2,.4,.6],xlabel='Brier score')
                ax.set_title(d.capitalize(),weight='bold',pad=8)
            else:
                ax.set(xlim=(-60,60),xticks=[-50,0,50],xlabel='Confidence gap (pp)')
                ax.axvline(0,ls='--',color='#555',lw=.8)
            clean(ax)
    fig.text(.19,.975,'Confidence in coefficient accuracy ($E_q<0.1$)',fontsize=9,weight='semibold',va='top')
    fig.text(.19,.915,'A  Brier score: lower is better',fontsize=8.5)
    fig.text(.19,.465,'B  Signed gap: negative = underconfident; positive = overconfident',fontsize=8.3)
    fig.text(.19,.025,'Points: run-balanced means. Bars: 95% intervals from resampling whole runs.\n'
             '10 supported runs per cell, except Gemini 3 Flash hard (9).',fontsize=8,linespacing=1.4)
    save(fig,'review_confidence')

def reliability_figure():
    fig,axs=plt.subplots(6,3,figsize=(6.75,7.6),sharex=True,sharey=True)
    fig.subplots_adjust(left=.15,right=.985,bottom=.12,top=.89,wspace=.25,hspace=.48)
    for i,m in enumerate(MODELS):
        for j,d in enumerate(DIFFS):
            ax=axs[i,j]
            ax.plot([0,1],[0,1],ls='--',lw=.6,color='#92989d')
            bs=[b for b in CELLS[m,d]['calibration_bins'] if b['n']]
            for b in bs:
                # Marker area gives support; hollow points flag especially sparse bins.
                ax.scatter(b['mean_confidence'],b['success_fraction'],s=10+.35*b['n'],
                           facecolor=MCOLORS[i] if b['n']>=5 else 'none',
                           edgecolor=MCOLORS[i],linewidth=.8,zorder=3)
            ax.set(xlim=(-.05,1.05),ylim=(-.05,1.05),xticks=[0,.5,1],yticks=[0,.5,1])
            ax.tick_params(labelsize=8,length=2)
            if i==0:ax.set_title(d.capitalize(),weight='bold',pad=17)
            if j==0:ax.set_ylabel(LABELS[i].replace(' ','\n',1),fontsize=8.3,labelpad=10)
            n=sum(b['n'] for b in bs)
            ax.text(.99,1.035,f'n = {n}',transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
            clean(ax,'both')
    fig.text(.57,.055,'Mean stated confidence within bin',ha='center',fontsize=8.5)
    fig.text(.018,.51,'Fraction with coefficient error $E_q<0.1$',rotation=90,va='center',fontsize=8.5)
    fig.text(.15,.98,'Descriptive reliability: report-weighted, not run-balanced',weight='semibold',fontsize=9,va='top')
    fig.text(.15,.95,'Populated bins; hollow markers have fewer than 5 coefficient observations.',fontsize=8)
    # Actual size key; areas use exactly the same rule as the plotted bins.
    hs=[plt.scatter([],[],s=10+.35*n,facecolors='none',edgecolors='#555',lw=.8) for n in [5,25,100]]
    fig.legend(hs,['5','25','100 coefficient observations'],loc='lower center',
               bbox_to_anchor=(.56,-.002),ncol=3,frameon=False,handletextpad=.8,columnspacing=1.4,fontsize=8)
    save(fig,'review_reliability')

def load_runs(path):
    manifest=json.loads((ROOT/'paper_artifacts/data_manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['archive_sha256']:
        raise ValueError('Run archive differs from pinned version 1 checksum')
    with zipfile.ZipFile(path) as z:
        runs=[json.loads(z.read(n)) for n in sorted(z.namelist()) if n.startswith('run_') and n.endswith('.json')]
    assert len(runs)==180
    for m in MODELS:
        for d in DIFFS:
            cell=[r for r in runs if r['config']['model']==m and r['config']['difficulty']==d]
            assert sorted(r['config']['seed'] for r in cell)==list(range(1,11))
    return runs

def load_local_runs(directory):
    """Verify the copied local records, without asserting a ZIP-byte checksum."""
    manifest=json.loads((ROOT/'data/local_run_manifest.json').read_text())
    for entry in manifest['files']:
        path=directory/entry['file']
        blob=path.read_bytes()
        assert len(blob)==entry['bytes'], path.name
        assert hashlib.sha256(blob).hexdigest()==entry['sha256'], path.name
    rr=[json.loads(p.read_text()) for p in sorted(directory.glob('run_*.json'))]
    assert len(rr)==180
    for m in MODELS:
        for d in DIFFS:
            cell=[r for r in rr if r['config']['model']==m and r['config']['difficulty']==d]
            assert sorted(r['config']['seed'] for r in cell)==list(range(1,11))
            scores=[r['results']['coefficient_errors']['total'] for r in cell]
            for threshold in [.01,.1]:
                assert sum(e<threshold for e in scores)==CELLS[m,d]['success'][str(threshold)]['count']
            for metric,source in [('solves','solve_count'),('checks','check_count'),('term_integrals','term_integral_count')]:
                values=[r['behavioral_metrics'].get(source,0) or 0 for r in cell]
                assert np.isclose(np.median(values),CELLS[m,d][metric]['median'])
                assert np.isclose(np.mean(values),CELLS[m,d][metric]['mean'])
    print('Local records: file hashes verified; 180 task cells and plotted success/action summaries agree.')
    return rr

def controls_figure():
    with gzip.open(ROOT/'paper_artifacts/control_baselines.json.gz','rt') as f:data=json.load(f)
    fig,axs=plt.subplots(1,3,figsize=(6.75,2.9),sharey=True)
    fig.subplots_adjust(left=.10,right=.99,bottom=.30,top=.80,wspace=.20)
    names={'fixed_sine':'Fixed sine','boundary_legendre':'Boundary Legendre','random_mixed':'Random mixed'}
    for ax,d in zip(axs,DIFFS):
        for k,(strategy,label) in enumerate(names.items()):
            c=next(r for r in data['summary'] if r['difficulty']==d and r['strategy']==strategy and r['grid_factor']==1)
            pts=c['checkpoints'];xs=[p['queries'] for p in pts]
            ys=[p['total_error']['success_e_lt_0.01']/p['total_error']['n'] for p in pts]
            ax.plot(xs,ys,color=COLORS[k],ls=['-','--',':'][k],marker=['o','s','^'][k],ms=3,lw=1.3,label=label)
            if strategy=='random_mixed':
                rr=[r for r in data['runs'] if r['difficulty']==d and r['strategy']==strategy and r['grid_factor']==1]
                byrep=np.array([[np.mean([r['checkpoints'][j]['coefficient_errors']['total']<.01 for r in rr if r['replicate']==rep]) for j in range(len(pts))] for rep in range(data['protocol']['random_replicates'])])
                ax.fill_between(xs,byrep.min(axis=0),byrep.max(axis=0),color=COLORS[k],alpha=.12,lw=0)
        ax.set(title=d.capitalize(),xlabel='Stored queries',ylim=(-.03,1.08),xticks=[xs[0],(xs[0]+xs[-1])//2,xs[-1]])
        clean(ax,'both')
    axs[0].set_ylabel('Success fraction ($E<0.01$)')
    fig.text(.10,.97,'Numerical query policies on the 30 reconstructed tasks',fontsize=9,weight='semibold',va='top')
    fig.text(.10,.88,'Shading: minimum–maximum across 10 random query lists, not a confidence interval.',fontsize=8)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.56,.015),ncol=3,frameon=False,fontsize=8)
    save(fig,'review_controls')

def numerics_figure():
    data=json.loads((ROOT/'paper_artifacts/identifiability_audit.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(6.75,3.15))
    fig.subplots_adjust(left=.11,right=.985,bottom=.22,top=.78,wspace=.40)
    rng=np.random.default_rng(20261102)
    for i,d in enumerate(DIFFS):
        vs=[r['all_stored_queries']['unit_column_norm']['min_over_max_singular_value'] for r in data['replayed_runs'] if r['difficulty']==d]
        axs[0].scatter(i+rng.uniform(-.15,.15,len(vs)),vs,s=9,alpha=.65,color=COLORS[i],linewidth=0)
        axs[0].plot([i-.22,i+.22],[np.median(vs)]*2,color='black',lw=1.4)
    axs[0].axhline(1e-8,color='#555',ls='--',lw=.8)
    axs[0].text(.035,1.6e-8,'Diagnostic cutoff $10^{-8}$',fontsize=8,
                bbox={'fc':'white','ec':'none','alpha':.9,'pad':.5})
    axs[0].set(yscale='log',ylabel=r'$\sigma_{\min}/\sigma_{\max}$ (unit columns)',
               xticks=range(3),xticklabels=[d.capitalize() for d in DIFFS],title='A  Numerical resolution')
    clean(axs[0],'y')
    for i,d in enumerate(DIFFS):
        rr=next(r for r in data['summary']['difficulty_groups'] if r['difficulty']==d)
        e=rr['amplitude_sensitivity'];xs=[r['common_amplitude'] for r in e];ys=[r['successes_at_001']/r['n'] for r in e]
        axs[1].plot(xs,ys,color=COLORS[i],marker=['o','s','^'][i],ms=4,lw=1.1,label=d.capitalize())
    axs[1].axvline(1,color='#555',ls=':',lw=.8,zorder=0)
    axs[1].text(1,.04,'Original scale',rotation=90,ha='right',va='bottom',fontsize=8,color='#444')
    axs[1].set(xscale='log',xticks=[1e-3,1,1e3,1e6],ylim=(0,1.04),
               xlabel='Common amplitude before\n12-decimal rounding',ylabel='Refit success fraction ($E<0.01$)',
               title='B  Sensitivity to amplitude')
    axs[1].legend(loc='lower right',frameon=False,fontsize=8);clean(axs[1],'both')
    fig.text(.11,.97,'Numerical checks on reconstructed archived measurements',fontsize=9,weight='semibold',va='top')
    fig.text(.11,.88,'60 query designs per setting; these are reconstructed fits, not new model runs.',fontsize=8)
    save(fig,'review_numerics')

def action_figure(runs):
    fig,axs=plt.subplots(3,3,figsize=(6.75,6.3),sharey=True)
    fig.subplots_adjust(left=.19,right=.99,bottom=.09,top=.91,wspace=.22,hspace=.40)
    measures=[('solve_count','Solves'),('check_count','Checks'),('term_integral_count','Term calls')]
    maxima={key:max(r['behavioral_metrics'].get(key,0) or 0 for r in runs) for key,_ in measures}
    rng=np.random.default_rng(20261102)
    for row,d in enumerate(DIFFS):
        for j,(key,label) in enumerate(measures):
            ax=axs[row,j]
            mx=0
            for i,m in enumerate(MODELS):
                vs=np.array([r['behavioral_metrics'].get(key,0) or 0 for r in runs if r['config']['model']==m and r['config']['difficulty']==d])
                mx=max(mx,vs.max())
                ax.scatter(vs,i+rng.uniform(-.16,.16,len(vs)),s=10,color=COLORS[j],alpha=.5,linewidth=0)
                ax.plot(np.median(vs),i,'|',color='black',ms=10,mew=1.5,zorder=4)
            ax.set(ylim=(5.55,-.55),yticks=range(6),xlim=(-.4,maxima[key]+1))
            if j==0:ax.set_yticklabels(LABELS)
            ax.set_title(f'{d.capitalize()} · {label}',fontsize=8.5,weight='semibold')
            clean(ax)
            if row==2:ax.set_xlabel('Calls per run')
    fig.text(.19,.98,'Tool use by action type',fontsize=10,weight='semibold',va='top')
    fig.text(.19,.945,'Dots: 10 runs per cell. Black ticks: medians. Action types are not stacked.',fontsize=8.3)
    save(fig,'review_actions')

def improvement_figure(runs):
    fig,axs=plt.subplots(3,1,figsize=(6.75,7.0),sharex=True)
    fig.subplots_adjust(left=.13,right=.985,bottom=.13,top=.92,hspace=.52)
    rng=np.random.default_rng(20261102)
    for row,(ax,d) in enumerate(zip(axs,DIFFS)):
        noniter=[]
        for i,m in enumerate(MODELS):
            rr=[r for r in runs if r['config']['model']==m and r['config']['difficulty']==d]
            vals=[];one=0;missing=0
            for r in rr:
                n=len(r.get('solves',[]));ratio=r['behavioral_metrics'].get('improvement_ratio')
                if n<=1:one+=1
                elif ratio is None:missing+=1
                else:vals.append(float(ratio))
            assert all(v>0 for v in vals)
            if vals:
                ax.scatter(i+rng.uniform(-.16,.16,len(vals)),vals,s=19,color=MCOLORS[i],alpha=.8,
                           edgecolors='white',linewidth=.4)
                med=np.median(vals);ax.plot([i-.23,i+.23],[med,med],color='#111',lw=1.5)
            noniter.append((one,missing,len(vals)))
            if row<2:ax.text(i,-.15,f'{len(vals)}/10',ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
        ax.axhline(1,color='#666',ls='--',lw=.7)
        ax.set_yscale('log');ax.set_ylabel('First / last solve error')
        ax.set_xticks(range(6),[f'{name}\n{n[2]}/10' for name,n in zip(SHORT,noniter)] if row==2 else SHORT)
        ax.set_xlim(-.5,5.5);clean(ax,'y')
        ax.set_title(d.capitalize()+' · runs with at least two solves and a recorded ratio',loc='left',fontsize=8.5,weight='semibold')
        print('improvement counts',d,noniter)
    fig.text(.13,.985,'Error change among repeated-solve runs',fontsize=10,weight='semibold',va='top')
    fig.text(.13,.95,'Dots: observed ratios. Black lines: medians of observed ratios. Dashed line: no error change.',fontsize=8)
    fig.text(.13,.025,'Counts below each panel show eligible runs / all 10 scored runs.\nThe 21 single-solve runs have no plotted ratio; all 159 repeated-solve ratios are available.',fontsize=8,linespacing=1.4)
    save(fig,'review_improvement')

def diversity_figure(runs):
    fig,axs=plt.subplots(1,3,figsize=(6.75,3.1),sharey=True)
    fig.subplots_adjust(left=.19,right=.99,bottom=.19,top=.77,wspace=.18)
    rng=np.random.default_rng(20261102)
    for j,(ax,d) in enumerate(zip(axs,DIFFS)):
        for i,m in enumerate(MODELS):
            vals=np.array([r['behavioral_metrics'].get('family_entropy_normalized',0) or 0 for r in runs if r['config']['model']==m and r['config']['difficulty']==d])
            q1,med,q3=np.quantile(vals,[.25,.5,.75])
            ax.scatter(vals,i+rng.uniform(-.15,.15,len(vals)),s=12,color=COLORS[j],alpha=.6,linewidth=0)
            ax.plot([q1,q3],[i,i],color='#222',lw=1.1)
            ax.plot(med,i,'|',color='#111',ms=10,mew=1.6)
        ax.set(xlim=(-.025,1.025),ylim=(5.55,-.55),yticks=range(6),xticks=[0,.5,1])
        if j==0:ax.set_yticklabels(LABELS)
        ax.set_title(d.capitalize(),weight='bold');ax.set_xlabel('Normalized entropy')
        clean(ax)
    fig.text(.19,.975,'Expression-family diversity',fontsize=10,weight='semibold',va='top')
    fig.text(.19,.885,'Dots: 10 runs per cell. Black mark: median and interquartile range.',fontsize=8.3)
    save(fig,'review_diversity')

def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--archive',type=Path)
    source.add_argument('--runs-dir',type=Path)
    args=parser.parse_args()
    task_figure();success_figure();confidence_figure();reliability_figure();controls_figure();numerics_figure()
    if args.archive:
        rr=load_runs(args.archive);action_figure(rr);improvement_figure(rr);diversity_figure(rr)
    elif args.runs_dir:
        rr=load_local_runs(args.runs_dir);action_figure(rr);improvement_figure(rr);diversity_figure(rr)
    else: print('Raw-run figures skipped: supply --archive or --runs-dir for actions, improvement, diversity.')

if __name__=='__main__':main()
