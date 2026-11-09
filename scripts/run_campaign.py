#!/usr/bin/env python3
"""Thin local CLI for original immutable 2026 B1/B3 training source modules.

NOT an evidence of identical GPU results: training requires the original HDF5 shards,
matching software/GPU environment and can be nondeterministic across systems.
"""
from pathlib import Path
import sys, argparse, json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))

parser=argparse.ArgumentParser(description='Run or validate a DAS B1/B3 campaign using original scientific code.')
parser.add_argument('--campaign',choices=('b1','b3'),required=True)
parser.add_argument('--mode',choices=('validate','smoke','full'),default='validate')
parser.add_argument('--phase2-root',type=Path,required=True,
                    help='Directory holding PHASE2_SHA256.txt, metadata/ and all 16 shards/')
parser.add_argument('--output',type=Path,help='Separate EMPTY experiment output directory (smoke/full only)')
args=parser.parse_args()
phase2=args.phase2_root.expanduser().resolve()
if not (phase2/'PHASE2_SHA256.txt').is_file(): parser.error('Missing Phase-2 fingerprint file')
if not (phase2/'metadata'/'SAMPLE_REGISTRY.csv').is_file(): parser.error('Missing Phase-2 sample registry')
if not (phase2/'shards').is_dir(): parser.error('Missing 16 Phase-2 HDF5 shards directory')

if args.campaign=='b1':
    from b1r_campaign.config import CampaignConfig
    from b1r_campaign.data import load_b1r_registry,assert_zero_cross_partition_raw_overlap,frames
    from b1r_campaign.campaign import run_smoke,run_full
    cfg=CampaignConfig()
    registry=load_b1r_registry(phase2,ROOT/'frozen')
    assert_zero_cross_partition_raw_overlap(registry)
    pieces=frames(registry)
else:
    from b3_spectral_campaign.config import Config
    from b3_spectral_campaign.data import load_registry,assert_target_isolation,frames
    from b3_spectral_campaign.campaign import smoke as run_smoke,full as run_full
    cfg=Config()
    registry=load_registry(phase2,ROOT/'frozen')
    assert_target_isolation(registry)
    pieces=frames(registry)
print(json.dumps({'campaign':args.campaign,'mode':args.mode,'patch_count':len(registry),
                  'phase2_root':str(phase2),'rows_by_role':{k:len(v) for k,v in pieces.items()}},indent=2))
if args.mode=='validate': sys.exit(0)
if args.output is None: parser.error('smoke/full requires --output')
out=args.output.expanduser().resolve()
# Keep the original Drive experiment folder untouched and avoid mixing completed runs.
if out==phase2 or out==ROOT or phase2 in out.parents or ROOT in out.parents:
    parser.error('--output must be separate from both the dataset and repository directories')
if args.mode=='smoke':
    out.mkdir(parents=True,exist_ok=True)
    run_smoke(pieces,phase2,cfg,out)
else:
    if out.exists() and any(out.iterdir()):
        parser.error('Full campaign output must initially be an empty directory. Never overwrite frozen runs.')
    out.mkdir(parents=True,exist_ok=True)
    run_full(pieces,phase2,cfg,out)
