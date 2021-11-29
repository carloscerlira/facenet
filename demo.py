from model import get_iimas_model
iimas_model = get_iimas_model(pretrained=True)

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from PIL import Image, ImageOps
import IPython
from imageio import imread
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

iimas_model = get_iimas_model()
iimas_model.eval()

#prepare preprocess pipeline
preprocess_pipelines = [transforms.Resize(224), 
                       transforms.CenterCrop(224), 
                       transforms.ToTensor(), 
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])]
trfrm = transforms.Compose(preprocess_pipelines)
topil = transforms.ToPILImage()
totensor = transforms.Compose(preprocess_pipelines[:-1])

''' 
----------------------------------------------------------------------------------------------------
                                Funcion para realizar autenicacion 

----------------------------------------------------------------------------------------------------
Params:         img_new:  La imagen en tiempo real
                url_person: url de la carpeta (incluye nombre de imagenes sin numero
                threshold a comparar'''
def auth_face(img_new, url_person, threshold, n=6):
  img1 = trfrm(Image.open(img_new)).unsqueeze(0)
  if url_person == '/content/drive/MyDrive/imagenes/carlos_cerritos/carlos_cerritos': n=4
  mages_paths = [url_person +'_0'+str(i)+'.jpg' for i in range(1,n)]
  distances = []

  for image_bd in mages_paths:
    img2 = trfrm(Image.open(image_bd)).unsqueeze(0)
    embed1, embed2 = iimas_model(img1), iimas_model(img2)
    euclidean_distance = F.pairwise_distance(embed1, embed2)
    distances.append(euclidean_distance)
  
  genuine = np.any(np.array(distances)<=threshold)
  return genuine

''' 
----------------------------------------------------------------------------------------------------
                            Creacion de embbeddings de imagenes existentes

----------------------------------------------------------------------------------------------------
'''
images_path = '/content/drive/MyDrive/imagenes'
people = ['artemio_padilla','carlos_cerritos','guillermo_cota','israel_cabello', 'jose_aguilar', 'pamela_ruiz','raul_mosqueda', 'patricio_barrero', 'luis_veleros']
#people = people[1:]

embeddings = defaultdict(list)
n=6
for p in people:
    if p == 'carlos_cerritos': n=4
    images_paths = [images_path+'/'+p+'/'+p+'_0'+str(i)+'.jpg' for i in range(1,n)]

    for ip in images_paths:
        img = Image.open(ip)
        img = ImageOps.exif_transpose(img)
        embeddings[p].append(iimas_model(trfrm(img).unsqueeze(0)))


''' 
----------------------------------------------------------------------------------------------------
                                Funcion para realizar recomocimirnto

----------------------------------------------------------------------------------------------------
Params:         img_name:  La imagen en tiempo real
                '''

src_path = '/content/drive/MyDrive/imagenes/unknow/'
def face_recognition(img_name,src_path=src_path, alpha = 0.5, embeddings = embeddings, mode = 1):
    img = Image.open(img_name)#src_path+img_name)
    img = ImageOps.exif_transpose(img)
    
    curr_emb = iimas_model(trfrm(img).unsqueeze(0))

    names = []

    for p in embeddings:
        for emb in embeddings[p]:

            d = F.pairwise_distance(curr_emb, emb)
            if d < alpha:
                names.append((p,d.item()))

    names.sort(key= lambda x:x[1])
    if names != []:
        if mode == 1:
            print(f'Welcome back {names[0][0]}, please make yourself confortable and do a lot of science!')
            return True
    else: 
        print('Face not recognized, please find yourself at the map in the entry, do not try to much or a T-800 will come to pick you up.')
        return False