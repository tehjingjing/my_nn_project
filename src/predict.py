"""Predict one new durian leaf image with a saved checkpoint."""
import argparse, os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import CustomCNN

def main():
    ap=argparse.ArgumentParser();ap.add_argument('image')
    ap.add_argument('--model',default=os.getenv('MODEL_PATH','/app/models/baseline/best_model.pth'))
    a=ap.parse_args();device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    ck=torch.load(a.model,map_location=device,weights_only=False);classes=ck['classes'];settings=ck.get('settings',{})
    model=CustomCNN(len(classes),settings.get('dropout',.3)).to(device);model.load_state_dict(ck['state_dict']);model.eval()
    tf=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    with Image.open(a.image) as img: x=tf(img.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad(): probs=F.softmax(model(x),dim=1)[0]
    order=probs.argsort(descending=True)
    print(f"Prediction: {classes[order[0]]} ({probs[order[0]].item():.4f})")
    for i in order: print(f"{classes[i]}: {probs[i].item():.4f}")

if __name__=='__main__':main()
