from model import get_iimas_model

import torch 
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

iimas_model = get_iimas_model(pretrained=True)
iimas_model.to(device)
iimas_model = torch.nn.DataParallel(iimas_model)
iimas_model.eval()

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

preprocess_pipelines = [transforms.Resize(224), 
                       transforms.CenterCrop(224), 
                       transforms.ToTensor(), 
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])]
trfrm = transforms.Compose(preprocess_pipelines)

import pandas as pd
import numpy as np 
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt 
plt.style.use("seaborn")

def get_face_classes(df):
    face_classes = dict()
    for idx, label in enumerate(df['class']):
        if label not in face_classes:
            face_classes[label] = []
        face_classes[label].append(df.iloc[idx, 0])
    return face_classes

def gen_triplets(df, face_classes, num_triplets, root_dir):
    triplets    = []
    classes     = df['class'].unique()
    
    for _ in range(num_triplets):
        pos_class = np.random.choice(classes)
        neg_class = np.random.choice(classes)
        while len(face_classes[pos_class]) < 2:
            pos_class = np.random.choice(classes)
        while pos_class == neg_class:
            neg_class = np.random.choice(classes)
        
        pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
        neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

        if len(face_classes[pos_class]) == 2:
            ianc, ipos = np.random.choice(2, size = 2, replace = False)
        else:
            ianc = np.random.randint(0, len(face_classes[pos_class]))
            ipos = np.random.randint(0, len(face_classes[pos_class]))
            while ianc == ipos:
                ipos = np.random.randint(0, len(face_classes[pos_class]))
        ineg = np.random.randint(0, len(face_classes[neg_class]))
        

        anc_id = face_classes[pos_class][ianc]
        pos_id = face_classes[pos_class][ipos]
        neg_id = face_classes[neg_class][ineg]

        anc_path  = root_dir+"/"+str(pos_name)+"/"+str(anc_id)+'.jpg'
        pos_path  = root_dir+"/"+str(pos_name)+"/"+str(pos_id)+'.jpg'
        neg_path  = root_dir+"/"+str(neg_name)+"/"+str(neg_id)+'.jpg'

        triplets.append([anc_path, pos_path, neg_path])

    return triplets

def gen_emb_triplets(triplets):
    emb_triplets = []
    for triplet in triplets:
        anc_path, pos_path, neg_path = triplet 
                
        anc = trfrm(Image.open(anc_path)).unsqueeze(0)
        pos = trfrm(Image.open(pos_path)).unsqueeze(0)
        neg = trfrm(Image.open(neg_path)).unsqueeze(0)
        
        anc_emb, pos_emb, neg_emb = iimas_model(anc), iimas_model(pos), iimas_model(neg)
        emb_triplets.append([anc_emb, pos_emb, neg_emb])
    return emb_triplets

def gen_scores(n_triplets, df, face_classes, root_dir, batch_size=1):
    triplets = gen_triplets(df, face_classes, n_triplets, root_dir)
    y_true, y_pred = [], []
    for i in range(0, n_triplets, batch_size):
        emb_triplets = gen_emb_triplets(triplets[i:i+batch_size])
        for emb_triplet in emb_triplets:            
            anc_emb, pos_emb, neg_emb = emb_triplet 
            
            pos_dis = F.pairwise_distance(anc_emb, pos_emb)
            neg_dis = F.pairwise_distance(anc_emb, neg_emb)
            
            y_true.append(1)
            y_pred.append(pos_dis.cpu().detach().numpy()[0])

            y_true.append(0)
            y_pred.append(neg_dis.cpu().detach().numpy()[0])
            
    return [y_true, y_pred] 

phase_dir = {'train': './datasets/train',
            'test': './datasets/test'}

phase_df = {'train': './logs/preprocess/train.csv', 
            'test': './logs/preprocess/test.csv'}

import pickle

def gen_scores_files(n_triplets=5):
    for phase in phase_dir:
        root_dir = phase_dir[phase]
        df = pd.read_csv(phase_df[phase])
        face_classes = get_face_classes(df)
        scores = gen_scores(n_triplets=n_triplets, df=df, face_classes=face_classes, root_dir=root_dir)

        filename = f'./logs/scores/scores_{phase}.obj'
        outfile = open(filename,'wb')
        pickle.dump(scores, outfile)
        outfile.close()
