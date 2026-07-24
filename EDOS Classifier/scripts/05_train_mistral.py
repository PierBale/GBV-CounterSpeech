#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import pandas as pd, torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
sys.path.append(str(Path(__file__).resolve().parents[1]/'src'))
from edos_khan.common import set_seed, device, TextDS, load_label_map, class_weights, metrics, save_preds

def eval_model(model,loader,dev):
    model.eval(); gold=[]; pred=[]; probs=[]
    with torch.no_grad():
        for b in loader:
            y=b.pop('labels').to(dev); b={k:v.to(dev) for k,v in b.items()}; o=model(**b); p=torch.softmax(o.logits.float(),dim=-1); pr=p.argmax(1)
            gold+=y.cpu().tolist(); pred+=pr.cpu().tolist(); probs+=p.cpu().numpy().tolist()
    return gold,pred,probs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',choices=['b','c'],required=True); ap.add_argument('--data-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--model-name',default='mistralai/Mistral-7B-v0.1'); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--max-length',type=int,default=150); ap.add_argument('--weight-decay',type=float,default=5e-3); ap.add_argument('--seed',type=int,default=2000); ap.add_argument('--selection-split',choices=['dev','test'],default='test'); ap.add_argument('--no-4bit',action='store_true'); args=ap.parse_args()
    set_seed(args.seed); dev=device(); data=Path(args.data_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(data/'train.csv'); ev=pd.read_csv(data/f'{args.selection_split}.csv'); te=pd.read_csv(data/'test.csv'); _,i2l=load_label_map(data/'label_map.json'); labels=[i2l[i] for i in range(len(i2l))]
    tok=AutoTokenizer.from_pretrained(args.model_name); tok.pad_token=tok.eos_token
    q=None if args.no_4bit else BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForSequenceClassification.from_pretrained(args.model_name,num_labels=len(labels),id2label={i:l for i,l in enumerate(labels)},label2id={l:i for i,l in enumerate(labels)},quantization_config=q,device_map='auto' if torch.cuda.is_available() else None)
    model.config.pad_token_id=tok.pad_token_id
    if q is not None: model=prepare_model_for_kbit_training(model)
    cfg=LoraConfig(task_type=TaskType.SEQ_CLS,r=16,lora_alpha=8,lora_dropout=0.05,bias='none',target_modules=['q_proj','k_proj','v_proj','o_proj']); model=get_peft_model(model,cfg); model.print_trainable_parameters()
    tr_loader=DataLoader(TextDS(tr.text,tr.label_id,tok,args.max_length),batch_size=args.batch_size,shuffle=True); ev_loader=DataLoader(TextDS(ev.text,ev.label_id,tok,args.max_length),batch_size=args.batch_size); te_loader=DataLoader(TextDS(te.text,te.label_id,tok,args.max_length),batch_size=args.batch_size)
    loss_fn=nn.CrossEntropyLoss(weight=class_weights(tr.label_id.tolist(),len(labels),dev)); opt=AdamW(model.parameters(),lr=args.lr,betas=(.9,.98),eps=1e-6,weight_decay=args.weight_decay); sched=get_linear_schedule_with_warmup(opt,0,args.epochs*len(tr_loader))
    best=-1; hist=[]
    for ep in range(1,args.epochs+1):
        model.train(); total=0
        for b in tqdm(tr_loader,desc=f'epoch {ep}/{args.epochs}'):
            y=b.pop('labels').to(dev); b={k:v.to(dev) for k,v in b.items()}; opt.zero_grad(); o=model(**b); loss=loss_fn(o.logits,y); loss.backward(); opt.step(); sched.step(); total+=loss.item()
        y,p,_=eval_model(model,ev_loader,dev); m=metrics(y,p,labels); hist.append({'epoch':ep,'selection_macro_f1':m['macro_f1'],'loss':total}); print(ep,m['macro_f1'])
        if m['macro_f1']>best: best=m['macro_f1']; model.save_pretrained(out/'best_adapter'); tok.save_pretrained(out/'best_adapter')
    y,p,prob=eval_model(model,te_loader,dev); m=metrics(y,p,labels); (out/'test_metrics.json').write_text(json.dumps(m,indent=2)); save_preds(out/'test_predictions.csv',te.text.tolist(),y,p,i2l,prob); (out/'training_history.json').write_text(json.dumps(hist,indent=2)); print(json.dumps({'test_macro_f1':m['macro_f1'],'best_selection_f1':best},indent=2))
if __name__=='__main__': main()
