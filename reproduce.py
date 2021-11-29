import argparse

parser = argparse.ArgumentParser()
args = {'train_dir':'./datasets/train',
        'test_dir':'./datasets/test',
        'train_csv':'./logs/preprocess/train.csv',
        'test_csv':'./logs/preprocess/test.csv',
        'num_epochs':2,
        'batch_size':1,
        'num_train_triplets':2,
        'num_test_triplets':2,
        'learning_rate':0.5,
        'step_size':50,
        'margin':0.5}

from model import get_iimas_model
from create_labels import create_labels
from metrics import gen_scores_files
from evaluate import gen_roc, gen_loss_curve
from train import train_model 

# REPRODUCE ROC
# this will generate auxiliar csv files found in logs/preprocess
create_labels(args)
# this will generate pickle score files found in logs/scores
gen_scores_files(n_triplets=5)
# this will generate the ROC curve found in logs/img/roc.png
gen_roc()

# TRAIN MODEL
# This one can take a few hours 
# train_model(args)
# this will generate loss curve found at logs/img/loss.png
# gen_loss_curve()

# To test our pretrained model with image 
model = get_iimas_model(pretrained=True)
img_path = ''
