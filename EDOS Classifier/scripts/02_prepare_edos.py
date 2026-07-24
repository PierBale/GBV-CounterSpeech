#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'src'))
from edos_khan.common import load_data, labels_for, make_maps, add_ids, save_label_map

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',choices=['b','c'],required=True); ap.add_argument('--edos-csv',required=True); ap.add_argument('--augmentation-csv'); ap.add_argument('--output-dir',required=True); ap.add_argument('--no-dev-in-train',action='store_true'); args=ap.parse_args()
    l2i,_=make_maps(labels_for(args.task))
    tr,dev,te=load_data(Path(args.edos_csv),args.task,Path(args.augmentation_csv) if args.augmentation_csv else None,include_dev=not args.no_dev_in_train)
    tr,dev,te=add_ids(tr,l2i),add_ids(dev,l2i),add_ids(te,l2i)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tr.to_csv(out/'train.csv',index=False); dev.to_csv(out/'dev.csv',index=False); te.to_csv(out/'test.csv',index=False); save_label_map(out/'label_map.json',l2i)
    print(f'Task {args.task.upper()} | train={tr.shape} dev={dev.shape} test={te.shape}')
    print(tr.label.value_counts())
if __name__=='__main__': main()
