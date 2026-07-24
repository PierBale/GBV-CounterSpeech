#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import pandas as pd, torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
sys.path.append(str(Path(__file__).resolve().parents[1]/'src'))
from edos_khan.common import set_seed, device, load_label_map, class_weights, metrics, save_preds
class DualDS(Dataset):
    def __init__(self,texts,labels,dtok,rtok,max_len): self.texts=list(texts); self.labels=list(labels); self.dtok=dtok; self.rtok=rtok; self.max_len=max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self,i):
        d=self.dtok(self.texts[i],max_length=self.max_len,padding='max_length',truncation=True,return_tensors='pt'); r=self.rtok(self.texts[i],max_length=self.max_len,padding='max_length',truncation=True,return_tensors='pt')
        return {'d_input_ids':d.input_ids.squeeze(0),'d_attention_mask':d.attention_mask.squeeze(0),'r_input_ids':r.input_ids.squeeze(0),'r_attention_mask':r.attention_mask.squeeze(0),'labels':torch.tensor(int(self.labels[i]))}
class DTFN(nn.Module):
    def __init__(self,dsrc,rsrc,n,dropout=.1):
        super().__init__(); self.d=AutoModel.from_pretrained(dsrc,output_hidden_states=True,ignore_mismatched_sizes=True); self.r=AutoModel.from_pretrained(rsrc,output_hidden_states=True,ignore_mismatched_sizes=True); self.drop=nn.Dropout(dropout); self.fc=nn.Linear(self.d.config.hidden_size+self.r.config.hidden_size,n)
    def forward(self,d_input_ids,d_attention_mask,r_input_ids,r_attention_mask):
        do=self.d(input_ids=d_input_ids,attention_mask=d_attention_mask).last_hidden_state[:,0,:]; ro=self.r(input_ids=r_input_ids,attention_mask=r_attention_mask).last_hidden_state[:,0,:]; return self.fc(self.drop(torch.cat([do,ro],dim=-1)))
def eval_model(model,loader,dev):
    model.eval(); gold=[]; pred=[]; probs=[]
    with torch.no_grad():
        for b in loader:
            y=b.pop('labels').to(dev); b={k:v.to(dev) for k,v in b.items()}; logits=model(**b); p=torch.softmax(logits,dim=-1); pr=p.argmax(1); gold+=y.cpu().tolist(); pred+=pr.cpu().tolist(); probs+=p.cpu().numpy().tolist()
    return gold,pred,probs
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',choices=['b','c'],required=True); ap.add_argument('--data-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--deberta-model',default='microsoft/deberta-v3-large'); ap.add_argument('--roberta-model',default='roberta-large'); ap.add_argument('--deberta-checkpoint'); ap.add_argument('--roberta-checkpoint'); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--lr',type=float,default=6e-6); ap.add_argument('--batch-size',type=int,default=4); ap.add_argument('--max-length',type=int,default=150); ap.add_argument('--weight-decay',type=float,default=5e-3); ap.add_argument('--seed',type=int,default=2000); ap.add_argument('--selection-split',choices=['dev','test'],default='test'); args=ap.parse_args()
    set_seed(args.seed); dev=device(); data=Path(args.data_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    tr=pd.read_csv(data/'train.csv'); ev=pd.read_csv(data/f'{args.selection_split}.csv'); te=pd.read_csv(data/'test.csv'); _,i2l=load_label_map(data/'label_map.json'); labels=[i2l[i] for i in range(len(i2l))]
    dtok=AutoTokenizer.from_pretrained(args.deberta_model); rtok=AutoTokenizer.from_pretrained(args.roberta_model); model=DTFN(args.deberta_checkpoint or args.deberta_model,args.roberta_checkpoint or args.roberta_model,len(labels)).to(dev)
    trl=DataLoader(DualDS(tr.text,tr.label_id,dtok,rtok,args.max_length),batch_size=args.batch_size,shuffle=True); evl=DataLoader(DualDS(ev.text,ev.label_id,dtok,rtok,args.max_length),batch_size=args.batch_size); tel=DataLoader(DualDS(te.text,te.label_id,dtok,rtok,args.max_length),batch_size=args.batch_size)
    loss_fn=nn.CrossEntropyLoss(weight=class_weights(tr.label_id.tolist(),len(labels),dev)); opt=AdamW(model.parameters(),lr=args.lr,betas=(.9,.98),eps=1e-6,weight_decay=args.weight_decay); sched=get_linear_schedule_with_warmup(opt,0,args.epochs*len(trl)); best=-1; hist=[]
    for ep in range(1,args.epochs+1):
        model.train(); total=0
        for b in tqdm(trl,desc=f'epoch {ep}/{args.epochs}'):
            y=b.pop('labels').to(dev); b={k:v.to(dev) for k,v in b.items()}; opt.zero_grad(); logits=model(**b); loss=loss_fn(logits,y); loss.backward(); opt.step(); sched.step(); total+=loss.item()
        y,p,_=eval_model(model,evl,dev); m=metrics(y,p,labels); hist.append({'epoch':ep,'selection_macro_f1':m['macro_f1'],'loss':total}); print(ep,m['macro_f1'])
        if m['macro_f1']>best: best=m['macro_f1']; torch.save({'model_state_dict':model.state_dict(),'best_f1':best,'args':vars(args)},out/'best_dtfn.pt')
    ck=torch.load(out/'best_dtfn.pt',map_location=dev); model.load_state_dict(ck['model_state_dict']); y,p,prob=eval_model(model,tel,dev); m=metrics(y,p,labels); (out/'test_metrics.json').write_text(json.dumps(m,indent=2)); save_preds(out/'test_predictions.csv',te.text.tolist(),y,p,i2l,prob); (out/'training_history.json').write_text(json.dumps(hist,indent=2)); print(json.dumps({'test_macro_f1':m['macro_f1'],'best_selection_f1':best},indent=2))
if __name__=='__main__': main()
