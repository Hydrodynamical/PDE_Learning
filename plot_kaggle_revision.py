"""Compact Kaggle-style figures, retaining the audited scientific values.

Usage: MPLCONFIGDIR=/private/tmp/pde-review-mpl python3 plot_kaggle_revision.py
No network calls or new model/control runs. Local records are hash-checked.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
IMG = ROOT / 'images'
MODELS = ['gemini-3.1-pro-preview', 'claude-opus-4-6', 'gpt-5.4',
          'gemini-3-flash-preview', 'claude-sonnet-4-20250514', 'gpt-5.4-mini']
LABELS = ['Gemini 3.1 Pro', 'Claude Opus 4.6', 'GPT-5.4',
          'Gemini 3 Flash', 'Claude Sonnet 4', 'GPT-5.4 mini']
SHORT = ['G3.1\nPro', 'Opus\n4.6', 'GPT\n5.4', 'G3\nFlash', 'Sonnet\n4', 'GPT\nmini']
FULL_TICKS = ['Gemini 3.1\nPro', 'Claude\nOpus 4.6', 'GPT-5.4',
              'Gemini 3\nFlash', 'Claude\nSonnet 4', 'GPT-5.4\nmini']
DIFFS = ['easy', 'medium', 'hard']
# Exact palettes used in the original Kaggle plotting scripts.
MODEL_COLORS = ['#4285F4', '#D97706', '#10B981', '#7BAAF7', '#F59E0B', '#6EE7B7']
DIFF_COLORS = ['#86efac', '#fbbf24', '#f87171']
ACTION_COLORS = ['#16a34a', '#f59e0b', '#06b6d4']
SUMMARY = json.loads((ROOT/'paper_artifacts/summary.json').read_text())
STATISTICS = json.loads((ROOT/'paper_artifacts/statistical_analysis.json').read_text())
CELLS = {(c['model'], c['difficulty']): c for c in SUMMARY['cells']}
STATS = {(c['model'], c['difficulty']): c for c in STATISTICS['cells']}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 10,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8.5, 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

def save(fig, name):
    fig.savefig(IMG/f'kaggle_{name}.pdf', metadata={
        'Author': 'Joseph K. Miller', 'CreationDate': None, 'ModDate': None})
    fig.savefig(IMG/f'kaggle_{name}.png', dpi=200)
    plt.close(fig)
    print(f'kaggle_{name}')

def clean(ax, axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color='#e8e8e8', lw=.55)
    ax.tick_params(length=2.5, width=.7)
    ax.spines['bottom'].set_linewidth(.7)
    ax.spines['left'].set_linewidth(.7)

def panels(title, height=3.35, bottom=.22, top=.82, sharey=True):
    fig, axs = plt.subplots(1, 3, figsize=(6.75, height), sharey=sharey)
    fig.subplots_adjust(left=.085, right=.99, bottom=bottom, top=top, wspace=.17)
    fig.suptitle(title, y=.98, fontsize=12, fontweight='bold')
    for ax,d in zip(axs, DIFFS):
        ax.set_title(d.capitalize(), fontweight='bold', pad=7)
        clean(ax)
    return fig, axs

def model_ticks(ax, counts=None):
    labels = SHORT if counts is None else [f'{s}\nn={n}' for s,n in zip(SHORT,counts)]
    ax.set_xticks(range(6), labels)
    ax.set_xlim(-.55,5.55)
    ax.tick_params(axis='x',length=0,pad=5)

def load_runs(directory):
    manifest=json.loads((ROOT/'data/local_run_manifest.json').read_text())
    for entry in manifest['files']:
        blob=(directory/entry['file']).read_bytes()
        assert len(blob)==entry['bytes']
        assert hashlib.sha256(blob).hexdigest()==entry['sha256']
    runs=[json.loads(p.read_text()) for p in sorted(directory.glob('run_*.json'))]
    assert len(runs)==180
    for m in MODELS:
        for d in DIFFS:
            rr=cell_runs(runs,m,d)
            assert sorted(r['config']['seed'] for r in rr)==list(range(1,11))
            assert sum(r['results']['coefficient_errors']['total']<.01 for r in rr)==CELLS[m,d]['success']['0.01']['count']
            for key,metric in [('solve_count','solves'),('check_count','checks'),('term_integral_count','term_integrals')]:
                vals=[r['behavioral_metrics'][key] for r in rr]
                assert np.isclose(np.mean(vals),CELLS[m,d][metric]['mean'])
    print('Verified local file hashes, 180 run records, and success/action summary agreement.')
    return runs

def cell_runs(runs,m,d):
    return [r for r in runs if r['config']['model']==m and r['config']['difficulty']==d]

def success_figure():
    fig,ax=plt.subplots(figsize=(6.75,3.25))
    fig.subplots_adjust(left=.09,right=.99,bottom=.20,top=.80)
    fig.suptitle('Benchmark success',y=.98,fontsize=12,fontweight='bold')
    x=np.arange(6); width=.27
    for j,d in enumerate(DIFFS):
        counts=[CELLS[m,d]['success']['0.01']['count'] for m in MODELS]
        bars=ax.bar(x+(j-1)*width,np.array(counts)/10,width*.92,
                    color=DIFF_COLORS[j],edgecolor='white',lw=.4,label=d.capitalize())
        for bar,n in zip(bars,counts):
            ax.text(bar.get_x()+bar.get_width()/2,max(bar.get_height(),0)+.025,
                    f'{n}/10',ha='center',va='bottom',fontsize=8,color='#303030')
    ax.set(xticks=x,xticklabels=FULL_TICKS,xlim=(-.55,5.55),ylim=(0,1.13),
           ylabel='Success rate ($E<0.01$)',yticks=[0,.2,.4,.6,.8,1])
    ax.tick_params(axis='x',length=0,pad=6)
    clean(ax)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,.91),
               ncol=3,frameon=False,columnspacing=2)
    save(fig,'success')

def actions_figure(runs):
    fig,axs=panels('Tool use',height=3.4,bottom=.25)
    keys=['solve_count','check_count','term_integral_count']
    names=['Solves','Checks','Term integrals']
    all_tops=[]
    for ax,d in zip(axs,DIFFS):
        bottom=np.zeros(6)
        for key,label,color in zip(keys,names,ACTION_COLORS):
            means=np.array([np.mean([r['behavioral_metrics'][key] for r in cell_runs(runs,m,d)]) for m in MODELS])
            ax.bar(range(6),means,bottom=bottom,width=.72,color=color,
                   edgecolor='white',lw=.4,label=label,alpha=.9)
            for i,v in enumerate(means):
                if v>=3.5:
                    ax.text(i,bottom[i]+v/2,f'{v:.1f}',ha='center',va='center',
                            fontsize=8,color='white',fontweight='bold')
            bottom+=means
        for i,total in enumerate(bottom):
            ax.text(i,total+.7,f'{total:.1f}',ha='center',va='bottom',fontsize=8,color='#444')
        all_tops.extend(bottom)
        model_ticks(ax)
    axs[0].set_ylabel('Mean actions per run')
    axs[0].set_ylim(0,max(all_tops)*1.14)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,.015),
               ncol=3,frameon=False,columnspacing=2)
    save(fig,'actions')

def improvement_figure(runs):
    fig,axs=panels('Within-run improvement',height=3.5,bottom=.245,top=.78)
    fig.text(.54,.888,'Bars: mean · black ticks and labels: median',ha='center',fontsize=8.5)
    rng=np.random.default_rng(42); n_total=0
    for ax,d in zip(axs,DIFFS):
        counts=[]
        for i,m in enumerate(MODELS):
            rr=cell_runs(runs,m,d)
            eligible=[r for r in rr if len(r.get('solves',[]))>=2]
            vals=np.array([r['behavioral_metrics']['improvement_ratio'] for r in eligible],dtype=float)
            assert len(vals)==len(eligible) and np.all(np.isfinite(vals)) and np.all(vals>0)
            counts.append(len(vals)); n_total+=len(vals)
            avg=float(np.mean(vals)); med=float(np.median(vals)); color=MODEL_COLORS[i]
            ax.bar(i,avg,width=.72,color=color,alpha=.82,edgecolor='white',lw=.5)
            ax.scatter(i+rng.uniform(-.20,.20,len(vals)),vals,s=10,color=color,alpha=.4,
                       edgecolors='none',zorder=4)
            ax.plot([i-.29,i+.29],[med,med],color='black',lw=1.2,zorder=5)
            label=f'{med:.1f}×' if med<10 else f'{med:.0f}×'
            ax.text(i,max(avg,med)*1.6,label,ha='center',va='bottom',fontsize=8,
                    color=color,fontweight='bold')
        ax.set(yscale='log',ylim=(1e-4,8e4),yticks=[1e-4,1e-2,1,1e2,1e4])
        ax.tick_params(axis='y',which='minor',length=0)
        ax.axhline(1,color='#777',ls=':',lw=.7,zorder=1)
        model_ticks(ax,counts)
    assert n_total==159
    axs[0].set_ylabel('First / last solve error')
    save(fig,'improvement')

def diversity_figure(runs):
    fig,axs=panels('Expression-family diversity',height=3.2,bottom=.19,top=.78)
    fig.text(.54,.885,'Bars: mean · black ticks and labels: median',ha='center',fontsize=8.5)
    rng=np.random.default_rng(42)
    for ax,d in zip(axs,DIFFS):
        for i,m in enumerate(MODELS):
            vals=np.array([r['behavioral_metrics']['family_entropy_normalized'] for r in cell_runs(runs,m,d)])
            assert len(vals)==10 and np.all(np.isfinite(vals))
            avg=float(np.mean(vals));med=float(np.median(vals));color=MODEL_COLORS[i]
            ax.bar(i,avg,width=.72,color=color,alpha=.82,edgecolor='white',lw=.5)
            ax.scatter(i+rng.uniform(-.20,.20,len(vals)),vals,s=10,color=color,alpha=.4,
                       edgecolors='none',zorder=4)
            ax.plot([i-.29,i+.29],[med,med],color='black',lw=1.2,zorder=5)
            ax.text(i,max(avg,med)+.045,f'{med:.2f}',ha='center',va='bottom',
                    fontsize=8,color=color,fontweight='bold')
        ax.set(ylim=(0,1.08),yticks=[0,.25,.5,.75,1])
        model_ticks(ax)
    axs[0].set_ylabel('Family entropy')
    save(fig,'diversity')

def reliability_figure():
    fig,axs=panels('Confidence calibration',height=3.5,bottom=.34,top=.82)
    for ax,d in zip(axs,DIFFS):
        ax.plot([0,1],[0,1],ls='--',color='#888',lw=.8,alpha=.7,zorder=1)
        for i,m in enumerate(MODELS):
            bins=[b for b in CELLS[m,d]['calibration_bins'] if b['n']]
            xs=[b['mean_confidence'] for b in bins];ys=[b['success_fraction'] for b in bins]
            ax.plot(xs,ys,color=MODEL_COLORS[i],lw=1.2,alpha=.9,zorder=2)
            for x,y,b in zip(xs,ys,bins):
                ax.plot(x,y,'o',ms=3.1,mew=.7,color=MODEL_COLORS[i],
                        mfc='white' if b['n']<5 else MODEL_COLORS[i],zorder=3)
        ax.set(xlim=(-.03,1.03),ylim=(-.03,1.05),xticks=[0,.5,1],yticks=[0,.25,.5,.75,1],
               xlabel='Stated confidence')
        clean(ax,'both')
    axs[0].set_ylabel('Fraction correct')
    handles=[Line2D([0],[0],color=c,lw=1.4,marker='o',ms=3.5,label=name)
             for c,name in zip(MODEL_COLORS,LABELS)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.015),
               ncol=3,frameon=False,columnspacing=1.6,handlelength=1.8)
    save(fig,'reliability')

def confidence_figure():
    fig,axs=plt.subplots(2,3,figsize=(6.75,4.6),sharey='row')
    fig.subplots_adjust(left=.09,right=.99,bottom=.12,top=.85,wspace=.17,hspace=.47)
    fig.suptitle('Confidence summaries',y=.98,fontsize=12,fontweight='bold')
    for j,d in enumerate(DIFFS):
        for row,key in enumerate(['brier_score','signed_calibration_in_the_large_gap']):
            ax=axs[row,j]
            values=np.array([STATS[m,d]['confidence']['run_balanced'][key]['estimate'] for m in MODELS])
            if row:values*=100
            ax.bar(range(6),values,width=.72,color=MODEL_COLORS,alpha=.85,edgecolor='white',lw=.5)
            model_ticks(ax);clean(ax)
            if row==0:ax.set_title(d.capitalize(),fontweight='bold',pad=8)
            else:ax.axhline(0,color='#555',lw=.7)
    axs[0,0].set(ylabel='Brier score',ylim=(0,.5),yticks=[0,.1,.2,.3,.4,.5])
    axs[1,0].set(ylabel='Confidence gap (pp)',ylim=(-50,50),yticks=[-50,-25,0,25,50])
    save(fig,'confidence')

def controls_figure():
    with gzip.open(ROOT/'paper_artifacts/control_baselines.json.gz','rt') as f:data=json.load(f)
    fig,axs=panels('Fixed query policies',height=3.15,bottom=.28,top=.81)
    names={'fixed_sine':'Fixed sine','boundary_legendre':'Boundary Legendre','random_mixed':'Random mixed'}
    colors=[MODEL_COLORS[0],MODEL_COLORS[1],MODEL_COLORS[2]]
    for ax,d in zip(axs,DIFFS):
        for k,(strategy,label) in enumerate(names.items()):
            c=next(r for r in data['summary'] if r['difficulty']==d and r['strategy']==strategy and r['grid_factor']==1)
            pts=c['checkpoints'];xs=[p['queries'] for p in pts]
            ys=[p['total_error']['success_e_lt_0.01']/p['total_error']['n'] for p in pts]
            ax.plot(xs,ys,color=colors[k],ls=['-','--',':'][k],marker=['o','s','^'][k],
                    ms=3,lw=1.3,label=label)
            if strategy=='random_mixed':
                rr=[r for r in data['runs'] if r['difficulty']==d and r['strategy']==strategy and r['grid_factor']==1]
                byrep=np.array([[np.mean([r['checkpoints'][j]['coefficient_errors']['total']<.01
                    for r in rr if r['replicate']==rep]) for j in range(len(pts))]
                    for rep in range(data['protocol']['random_replicates'])])
                ax.fill_between(xs,byrep.min(axis=0),byrep.max(axis=0),color=colors[k],alpha=.12,lw=0)
        ax.set(xlabel='Stored queries',ylim=(-.03,1.08),xticks=[xs[0],(xs[0]+xs[-1])//2,xs[-1]],
               yticks=[0,.25,.5,.75,1])
        clean(ax,'both')
    axs[0].set_ylabel('Success rate ($E<0.01$)')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.015),
               ncol=3,frameon=False,columnspacing=1.6)
    save(fig,'controls')

def numerics_figure():
    data=json.loads((ROOT/'paper_artifacts/identifiability_audit.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(6.75,3.15))
    fig.subplots_adjust(left=.095,right=.985,bottom=.23,top=.79,wspace=.36)
    fig.suptitle('Numerical sensitivity',y=.98,fontsize=12,fontweight='bold')
    vals=[[r['all_stored_queries']['unit_column_norm']['min_over_max_singular_value']
           for r in data['replayed_runs'] if r['difficulty']==d] for d in DIFFS]
    bp=axs[0].boxplot(vals,positions=range(3),widths=.52,patch_artist=True,whis=1.5,
        medianprops={'color':'black','lw':1.2},whiskerprops={'lw':.9},capprops={'lw':.9},
        flierprops={'marker':'o','markersize':2.5,'markerfacecolor':'#999','markeredgewidth':0,'alpha':.5})
    for patch,color in zip(bp['boxes'],DIFF_COLORS):patch.set(facecolor=color,edgecolor='#555',alpha=.8,lw=.8)
    axs[0].axhline(1e-8,color='#777',ls=':',lw=.8)
    axs[0].set(yscale='log',xticks=range(3),xticklabels=[d.capitalize() for d in DIFFS],
               ylabel=r'$\sigma_{\min}/\sigma_{\max}$',title='A  Singular values')
    axs[0].tick_params(axis='y',which='minor',length=0)
    clean(axs[0])
    for i,d in enumerate(DIFFS):
        rr=next(r for r in data['summary']['difficulty_groups'] if r['difficulty']==d)
        ss=rr['amplitude_sensitivity'];xs=[r['common_amplitude'] for r in ss]
        ys=[r['successes_at_001']/r['n'] for r in ss]
        axs[1].plot(xs,ys,color=[MODEL_COLORS[2],MODEL_COLORS[1],'#e15d5d'][i],
                    marker=['o','s','^'][i],ms=3.8,lw=1.3,label=d.capitalize())
    axs[1].axvline(1,color='#888',ls=':',lw=.7)
    axs[1].set(xscale='log',xticks=[1e-3,1,1e3,1e6],ylim=(0,1.04),
               xlabel='Common amplitude',ylabel='Refit success rate',title='B  Amplitude scaling')
    axs[1].legend(loc='lower right',frameon=False,fontsize=8)
    clean(axs[1],'both')
    save(fig,'numerics')

def main():
    parser=argparse.ArgumentParser()
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument('--runs-dir',type=Path,default=ROOT/'data/recorded_runs')
    inputs.add_argument('--archive',type=Path,help='Pinned Kaggle dataset version 1 ZIP')
    parser.add_argument('--figure',choices=['success','actions','improvement','diversity','reliability','confidence','controls','numerics'])
    args=parser.parse_args()
    functions={'success':success_figure,'reliability':reliability_figure,'confidence':confidence_figure,
               'controls':controls_figure,'numerics':numerics_figure}
    raw={'actions':actions_figure,'improvement':improvement_figure,'diversity':diversity_figure}
    if args.figure in functions:functions[args.figure]();return
    if args.archive:
        from analyze_review_runs import load_archive
        manifest=json.loads((ROOT/'paper_artifacts/data_manifest.json').read_text())
        runs=load_archive(args.archive,manifest)
    else:
        runs=load_runs(args.runs_dir)
    if args.figure in raw:raw[args.figure](runs);return
    for fn in functions.values():fn()
    for fn in raw.values():fn(runs)

if __name__=='__main__':main()
