#!/usr/bin/env python3
from pathlib import Path
import argparse, json
from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-preds',nargs='+',required=True); ap.add_argument('--fallback-preds',required=True); ap.add_argument('--output-csv',required=True); ap.add_argument('--metrics-json',required=True); args=ap.parse_args()
    bases=[pd.read_csv(p) for p in args.base_preds]; fb=pd.read_csv(args.fallback_preds); rows=[]
    for i in range(len(fb)):
        votes=[b.loc[i,'label_pred'] for b in bases]; c=Counter(votes).most_common(); tied=(len(c)>1 and c[0][1]==c[1][1]); pred=fb.loc[i,'label_pred'] if tied else c[0][0]
        rows.append({'text':fb.loc[i,'text'],'gold_label':fb.loc[i,'gold_label'],'ensemble_pred':pred,'used_fallback':tied,'fallback_pred':fb.loc[i,'label_pred'],'base_votes':json.dumps(votes,ensure_ascii=False)})
    out=pd.DataFrame(rows); Path(args.output_csv).parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output_csv,index=False)
    y=out.gold_label.tolist(); p=out.ensemble_pred.tolist(); m={'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro')),'weighted_f1':float(f1_score(y,p,average='weighted')),'fallback_used_count':int(out.used_fallback.sum()),'fallback_used_rate':float(out.used_fallback.mean()),'confusion_matrix':confusion_matrix(y,p).tolist(),'classification_report':classification_report(y,p,zero_division=0,output_dict=True),'base_prediction_files':args.base_preds,'fallback_prediction_file':args.fallback_preds}
    Path(args.metrics_json).parent.mkdir(parents=True,exist_ok=True); Path(args.metrics_json).write_text(json.dumps(m,ensure_ascii=False,indent=2)); print(json.dumps(m,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
