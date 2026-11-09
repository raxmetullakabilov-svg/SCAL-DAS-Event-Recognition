from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

_METADATA={'Software':'phase2_compact'}

def _finish(fig, out):
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(out,dpi=140,metadata=_METADATA)
    plt.close(fig)
    return out


def plot_sample_count_by_class_split(samples: pd.DataFrame, out: Path) -> Path:
    tab=pd.crosstab(samples['class_name'],samples['b1_role']).sort_index()
    fig,ax=plt.subplots(figsize=(8,4.5))
    tab.plot(kind='bar',ax=ax)
    ax.set_xlabel('Class'); ax.set_ylabel('Samples'); ax.set_title('Sample count by class and B1 split')
    return _finish(fig,out)


def plot_event_to_sample_distribution(samples: pd.DataFrame, out: Path) -> Path:
    counts=samples.groupby('event_id').size().sort_values().reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(7,4.2))
    ax.plot(range(len(counts)),counts.to_numpy(),marker='o',linestyle='none')
    ax.set_xlabel('Event rank'); ax.set_ylabel('Samples per event'); ax.set_title('Event-to-sample distribution')
    return _finish(fig,out)


def plot_signal_value_diagnostics(signal: pd.DataFrame, out: Path) -> Path:
    s=signal.sort_values('sample_id',kind='stable').reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(8,4.2))
    x=range(len(s))
    ax.plot(x,s['raw_min'],label='raw_min')
    ax.plot(x,s['raw_max'],label='raw_max')
    ax.plot(x,s['model_min'],label='model_min')
    ax.plot(x,s['model_max'],label='model_max')
    ax.set_xlabel('Sample index'); ax.set_ylabel('Value'); ax.set_title('Signal value diagnostics'); ax.legend()
    return _finish(fig,out)
