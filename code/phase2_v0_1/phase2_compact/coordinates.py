from dataclasses import dataclass
from typing import Any
from .discovery import RawFileStructure

@dataclass(frozen=True)
class WindowGeometry:
    window_length: int
    hop: int
    time_axis: int
    channel_axis: int
    source_method: str

@dataclass(frozen=True)
class RawSupport:
    time_start: int
    time_end: int
    channel_start: int
    channel_end: int


def _first(attrs: dict[str, Any], names: tuple[str, ...]):
    for n in names:
        if n in attrs:
            return attrs[n]
    return None


def resolve_window_geometry(structure: RawFileStructure, override: dict[str, Any] | None = None) -> WindowGeometry:
    attrs = dict(structure.attributes)
    if override:
        attrs.update(override)
    wl = _first(attrs, ('window_length','window_size','samples_per_window'))
    hop = _first(attrs, ('hop','window_shift','shift','hop_length'))
    ta = _first(attrs, ('time_axis',))
    if wl is None or hop is None:
        raise ValueError('window length and hop are required to convert Phase-1 windows to raw samples')
    if ta is None:
        raise ValueError('time_axis must be explicit in source metadata or reviewed override')
    wl, hop, ta = int(wl), int(hop), int(ta)
    if wl <= 0 or hop <= 0:
        raise ValueError('window length and hop must be positive')
    if ta not in (0,1):
        raise ValueError('time_axis must be 0 or 1 for 2-D raw data')
    return WindowGeometry(wl, hop, ta, 1-ta, 'override+metadata' if override else 'metadata')


def event_to_raw_support(event, geom: WindowGeometry, raw_shape: tuple[int,int]) -> RawSupport:
    ws, we = int(event['window_start']), int(event['window_end'])
    cs, ce = int(event['channel_start']), int(event['channel_end'])
    if ws < 0 or we < ws or cs < 0 or ce < cs:
        raise ValueError('invalid frozen event geometry')
    t0 = ws * geom.hop
    t1 = we * geom.hop + geom.window_length
    c0, c1 = cs, ce + 1
    tdim = int(raw_shape[geom.time_axis])
    cdim = int(raw_shape[geom.channel_axis])
    if t0 < 0 or t1 > tdim:
        raise ValueError(f'time support [{t0},{t1}) outside raw time dimension {tdim}')
    if c0 < 0 or c1 > cdim:
        raise ValueError(f'channel support [{c0},{c1}) outside raw channel dimension {cdim}')
    return RawSupport(t0,t1,c0,c1)
