import json
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms
import glob
import shutil
from efficientnet_pytorch import EfficientNet
model_name = 'efficientnet-b3'



model = EfficientNet.from_name(model_name)
model._fc = nn.Linear(model._fc.in_features,3)
model.cuda()
checkpoint = torch.load("model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
model.eval()
image_size = EfficientNet.get_image_size(model_name) # 224

# Preprocess image
tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

img_paths = glob.glob("./split/tosplit/images/*.jpg")
map_dict = ["bg","blue","red"]
for img_path in img_paths:
    img = Image.open(img_path)
    img = tfms(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(img.cuda())
        prob = torch.softmax(logits, dim=1)
        idx = torch.argmax(prob,1).cpu().item()
        img_name = img_path.split("/")[-1]
        shutil.copy(img_path,"./split/{}/{}".format(map_dict[idx],img_name))
    
        
