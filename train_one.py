#!/usr/bin/env python3
"""Train a single algorithm. Called as subprocess to isolate CUDA state."""
import argparse, json, os

os.chdir("/home/jovyan/Finetunehub")

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", required=True)
parser.add_argument("--output_dir", required=True)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--config", help="Path to JSON file with kwargs")
group.add_argument("--kwargs_json", help="JSON string with kwargs")
args = parser.parse_args()

if args.config:
    with open(args.config) as f:
        kwargs = json.load(f)
else:
    kwargs = json.loads(args.kwargs_json)

from aligntune.core.backend_factory import create_rl_trainer

trainer = create_rl_trainer(
    algorithm=args.algorithm,
    output_dir=args.output_dir,
    **kwargs,
)
trainer.train()
print(f"Done: {args.output_dir}")
