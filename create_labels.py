import os
import glob
import pandas as pd
import time
from PIL import Image

def create_labels(args):
    for phase in ['train', 'test']:
        root_dir = args[f'{phase}_dir']
        files = glob.glob(root_dir+"/*/*")

        data = []
        for idx, file in enumerate(files):
            face_id    = os.path.basename(file).split(".")[0]
            face_label = os.path.basename(os.path.dirname(file))
            data.append([face_id, face_label])
        
        df = pd.DataFrame(data=data, columns=['id', 'name'])
        df['class'] = pd.factorize(df['name'])[0]
        df.to_csv(f'./logs/preprocess/{phase}.csv', index = False)