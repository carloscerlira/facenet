import argparse
import datetime
import time

import numpy as np
import pandas as pd 
import torch
import torch.optim as optim
from torch.nn.modules.distance import PairwiseDistance
from torch.optim import lr_scheduler

from data_loader import get_dataloader
from loss import TripletLoss
from model import FaceNetModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
l2_dist = PairwiseDistance(2)

def train_model(args):
    model = FaceNetModel()
    model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=0.1)
    triplet_loss = TripletLoss(args['margin']).to(device)

    loss = []
    for epoch in range(args['num_epochs']):
        print(80 * '=')
        print(f'Epoch: {epoch}')

        time0 = time.time()
        data_loaders, data_size = get_dataloader(args['train_dir'], args['test_dir'],
                                                 args['train_csv'], args['test_csv'],
                                                 args['num_train_triplets'], args['num_test_triplets'],
                                                 args['batch_size'])

        avg_triplet_loss = train_valid(args, model, optimizer, triplet_loss, scheduler, epoch, data_loaders, data_size)
        loss.append(avg_triplet_loss)
        print(f'  Execution time                 = {time.time() - time0}')
    
    loss_df = pd.DataFrame(data=loss, columns=['loss'])
    loss_df.to_csv("logs/scores/loss.csv")
    return 


def save_last_checkpoint(state):
    torch.save(state, 'log/last_checkpoint.pth')


def train_valid(args, model, optimizer, triploss, scheduler, epoch, dataloaders, data_size):
    for phase in ['train', 'valid']:

        labels, distances = [], []
        triplet_loss_sum = 0.0

        if phase == 'train':
            scheduler.step()
            if scheduler.last_epoch % scheduler.step_size == 0:
                print("LR decayed to:", ', '.join(map(str, scheduler.get_lr())))
            model.train()
        else:
            model.eval()

        for batch_idx, batch_sample in enumerate(dataloaders[phase]):

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)

            with torch.set_grad_enabled(phase == 'train'):

                anc_embed, pos_embed, neg_embed = model(anc_img), model(pos_img), model(neg_img)

                pos_dist = l2_dist.forward(anc_embed, pos_embed)
                neg_dist = l2_dist.forward(anc_embed, neg_embed)

                all = (neg_dist - pos_dist < args['margin']).cpu().numpy().flatten()
                if phase == 'train':
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                else:
                    hard_triplets = np.where(all >= 0)

                anc_hard_embed = anc_embed[hard_triplets]
                pos_hard_embed = pos_embed[hard_triplets]
                neg_hard_embed = neg_embed[hard_triplets]

                anc_hard_img = anc_img[hard_triplets]
                pos_hard_img = pos_img[hard_triplets]
                neg_hard_img = neg_img[hard_triplets]

                model.module.forward_classifier(anc_hard_img)
                model.module.forward_classifier(pos_hard_img)
                model.module.forward_classifier(neg_hard_img)

                triplet_loss = triploss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

                if phase == 'train':
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    optimizer.step()

                distances.append(pos_dist.data.cpu().numpy())
                labels.append(np.ones(pos_dist.size(0)))

                distances.append(neg_dist.data.cpu().numpy())
                labels.append(np.zeros(neg_dist.size(0)))

                triplet_loss_sum += triplet_loss.item()

        avg_triplet_loss = triplet_loss_sum / data_size[phase]
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        
        return avg_triplet_loss
