from dataclasses import dataclass, replace
import hashlib, math
import numpy as np
from .coordinates import WindowGeometry, RawSupport
from .config import Phase2Config

@dataclass(frozen=True)
class SamplePlan:
    sample_id: str
    event_id: str
    anchor_window: int
    request_time_start: int
    request_time_end: int
    source_time_start: int
    source_time_end: int
    request_channel_start: int
    request_channel_end: int
    source_channel_start: int
    source_channel_end: int
    pad_time_left: int
    pad_time_right: int
    pad_channel_left: int
    pad_channel_right: int
    time_axis: int


def make_sample_id(plan: SamplePlan) -> str:
    payload = '|'.join(map(str, [
        plan.event_id,
        plan.request_time_start, plan.request_time_end,
        plan.request_channel_start, plan.request_channel_end,
    ]))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]


def _crop_bounds(center: int, width: int, dim: int):
    start = int(center) - width // 2
    end = start + width
    src0=max(0,start); src1=min(dim,end)
    return start,end,src0,src1,src0-start,end-src1


def plan_event_samples(event, support: RawSupport, geom: WindowGeometry, raw_shape: tuple[int,int], cfg: Phase2Config) -> list[SamplePlan]:
    ws, we = int(event['window_start']), int(event['window_end'])
    stride_windows = max(1, int(math.ceil(cfg.model_time / geom.hop)))
    anchors = list(range(ws, we+1, stride_windows))
    if anchors[-1] != we:
        anchors.append(we)
    tdim=int(raw_shape[geom.time_axis]); cdim=int(raw_shape[geom.channel_axis])
    ccenter=(int(event['channel_start']) + int(event['channel_end'])) // 2
    rc0,rc1,sc0,sc1,pcl,pcr = _crop_bounds(ccenter, cfg.model_channels, cdim)
    plans=[]; seen=set()
    for w in anchors:
        anchor_center = w*geom.hop + geom.window_length//2
        rt0,rt1,st0,st1,ptl,ptr = _crop_bounds(anchor_center, cfg.model_time, tdim)
        key=(rt0,rt1,rc0,rc1)
        if key in seen:
            continue
        seen.add(key)
        p=SamplePlan('', str(event['event_id']), w, rt0,rt1,st0,st1,rc0,rc1,sc0,sc1,ptl,ptr,pcl,pcr,geom.time_axis)
        plans.append(replace(p, sample_id=make_sample_id(p)))
    return plans


def extract_fixed_patch(raw_dataset, plan: SamplePlan, cfg: Phase2Config) -> np.ndarray:
    if plan.time_axis == 0:
        x=np.asarray(raw_dataset[plan.source_time_start:plan.source_time_end,
                                 plan.source_channel_start:plan.source_channel_end])
    else:
        x=np.asarray(raw_dataset[plan.source_channel_start:plan.source_channel_end,
                                 plan.source_time_start:plan.source_time_end]).T
    x=np.pad(x, ((plan.pad_time_left,plan.pad_time_right),(plan.pad_channel_left,plan.pad_channel_right)), mode='constant')
    if x.shape != (cfg.model_time,cfg.model_channels):
        raise ValueError(f'fixed patch shape mismatch: {x.shape} != {(cfg.model_time,cfg.model_channels)}')
    return x
