"""Command-line experiment runner for the official OPSSAT-AD feature CSV."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.pipeline import run_experiments

p=argparse.ArgumentParser()
p.add_argument('--csv',default=str(ROOT / 'data' / 'dataset.csv'))
p.add_argument('--label',default=None)
p.add_argument('--split',default=None)
p.add_argument('--no-tune',action='store_true')
p.add_argument('--output',default=str(ROOT / 'results' / 'latest_experiment.json'))
a=p.parse_args()
df=pd.read_csv(a.csv)
result=run_experiments(df,a.label,a.split,tune=not a.no_tune)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
print(json.dumps({'output':str(out),'best_model':result['best_model'],'metrics':result['metrics']},indent=2))
