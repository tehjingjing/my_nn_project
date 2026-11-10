"""Evaluate one saved checkpoint on the held-out test split."""
import argparse, json, os, csv
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CustomCNN

MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
PROJECT_DIR=Path(__file__).resolve().parents[1]

def scores(labels, preds, classes):
    n=len(classes); cm=[[0]*n for _ in range(n)]
    for y,p in zip(labels,preds): cm[y][p]+=1
    report={}; total=len(labels); correct=sum(cm[i][i] for i in range(n)); wf=0
    for i,name in enumerate(classes):
        tp=cm[i][i]; support=sum(cm[i]); predicted=sum(row[i] for row in cm)
        precision=tp/predicted if predicted else 0.0; recall=tp/support if support else 0.0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        wf+=f1*support
        report[name]={"precision":precision,"recall":recall,"f1":f1,"support":support}
    return {"accuracy":correct/total,"weighted_f1":wf/total,"classes":report,"confusion_matrix":cm}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.getenv('DURIAN_DATA_DIR',str(PROJECT_DIR/'data'/'durian_leaf_dataset')))
    ap.add_argument('--model', default=os.getenv('MODEL_PATH',str(PROJECT_DIR/'models'/'baseline'/'best_model.pth')))
    ap.add_argument('--output-dir', default=os.getenv('OUTPUT_DIR',str(PROJECT_DIR/'models'/'evaluation')))
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint=torch.load(args.model,map_location=device,weights_only=False)
    classes=checkpoint['classes']; dropout=checkpoint.get('settings',{}).get('dropout',.3)
    ds=datasets.ImageFolder(Path(args.data_dir)/'test',transform=transforms.Compose([
        transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)]))
    if ds.classes!=classes: raise ValueError(f"Test class folders {ds.classes} don't match checkpoint classes {classes}")
    loader=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
    model=CustomCNN(len(classes),dropout).to(device); model.load_state_dict(checkpoint['state_dict']); model.eval()
    ys=[]; ps=[]; confs=[]
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device)); probs=F.softmax(logits,dim=1); p=probs.argmax(1)
            ys.extend(y.tolist()); ps.extend(p.cpu().tolist()); confs.extend(probs.max(1).values.cpu().tolist())
    result=scores(ys,ps,classes)
    train_counts=checkpoint.get('train_class_counts')
    if train_counts is None: raise ValueError('Checkpoint lacks training class counts; retrain with the current src/train.py')
    majority=max(range(len(classes)),key=lambda i:train_counts[i])
    result['majority_class_baseline']={"class":classes[majority],"accuracy":sum(y==majority for y in ys)/len(ys)}
    result['test_samples']=len(ds); result['device']=str(device)
    wrong=[]
    for i,(y,p) in enumerate(zip(ys,ps)):
        if y!=p: wrong.append({"image":str(ds.samples[i][0]),"actual":classes[y],"predicted":classes[p],"confidence":confs[i]})
    result['wrong_prediction_count']=len(wrong)
    (out/'test_results.json').write_text(json.dumps(result,indent=2))
    with (out/'wrong_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['image','actual','predicted','confidence']);w.writeheader();w.writerows(wrong)
    print(json.dumps(result,indent=2)); print(f"Wrong predictions saved to {out/'wrong_predictions.csv'}")
    if len(wrong)<5: print('Fewer than five wrong predictions; inspect all listed errors.')

if __name__=='__main__': main()
