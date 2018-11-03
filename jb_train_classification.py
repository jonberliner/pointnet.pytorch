import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from jb_pointnet import PointNetCls
from lsk.data.datasets import VariantDataset


parser = argparse.ArgumentParser()
# TODO: should be info in the dataset
parser.add_argument('--dim_input', type=int, help='input size')

parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--num_epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--output_dir', type=str, default='cls',  help='output folder')
parser.add_argument('--model_checkpoint', type=str, default='',  help='model path')

parser.add_argument('--random_seed', type=int, default=-1,  help='experiment seed')

opt = parser.parse_args()

try:
    os.makedirs(opt.output_dir,
                exist_ok=True)
except OSError:
    pass

if opt.random_seed < 0:
    opt.random_seed = random.randint(1e8, 1e9-1)

print (opt)

torch.manual_seed(opt.random_seed)

# BUILD DATASET
TRAINING_UIDS = ...
TESTING_UIDS = ...


def lookup_embedding(variant_info):
    attributes = extract_attributes(variant_info)


dataset = VariantDataset(variant_uids=TRAINING_UIDS,
                         transforms=transform,
                         path=args.data_path,
                         download=True)
data_loader = DataLoader(dataset,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         num_workers=int(opt.workers))

test_dataset = VariantDataset(variant_uids=TRAINING_UIDS,
                              transforms=transform,
                              path=args.test_data_path,
                              download=True)
test_data_loader = DataLoader(dataset,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=int(opt.workers))


# GPU training?
CUDA = torch.cuda.is_available()
device = 'cuda' if CUDA else 'cpu'

# BUILD MODEL
classifier = PointNetCls(dim_input=dim_input,
                         num_classes=num_classes)\
                        .to(device)

# (maybe) load checkpoint
if opt.model_checkpoint != '':
    classifier.load_state_dict(torch.load(opt.model_checkpoint))

# instantiate optimizer
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

# TRAINING LOOP
num_batch = len(dataset) // opt.batchSize
for epoch in range(opt.nepoch):
    for step_in_epoch, data in enumerate(data_loader, 0):

        # get data
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)

        # forward pass
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, _ = classifier(points)
        # calculate loss
        loss = F.nll_loss(pred, target)
        # backward pass
        loss.backward()
        # update
        optimizer.step()
        # print step performance
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %
              (epoch,
               step_in_epoch,
               num_batch,
               loss.item(),correct.item() / float(opt.batch_size)))

        if step_in_epoch % 10 == 0:
            step_in_test, data = next(enumerate(test_data_loader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = point.to(device), target.to(device)
            classifier = classifier.eval()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %
                  (epoch,
                   step_in_test,
                   num_batch,
                   blue('test'),
                   loss.item(),
                   correct.item()/float(opt.batch_size)))

    torch.save(classifier.state_dict(),
               '%s/cls_model_%d.pth' % (opt.output_dir, epoch))
    torch.save(embeddings.state_dict(),
               '%s/cls_embeddings_%d.pth' % (opt.output_dir, epoch))

