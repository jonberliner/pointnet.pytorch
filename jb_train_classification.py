import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from jb_pointnet import PointNetCls

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

if opt.random_seed < 1:
    opt.random_seed = random.randint(1e8, 1e9-1)

print (opt)

# blue = lambda x:'\033[94m' + x + '\033[0m'

torch.manual_seed(opt.random_seed)

dataset = ...
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=int(opt.workers))

test_dataset = ...
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=int(opt.workers))


CUDA = torch.cuda.is_available()
device = 'cuda' if CUDA else 'cpu'

classifier = PointNetCls(dim_input=dim_input,
                         num_classes=num_classes)
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

if opt.model_checkpoint != '':
    classifier.load_state_dict(torch.load(opt.model_checkpoint))

classifier.to(device)

num_batch = len(dataset) // opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)

        optimizer.zero_grad()
        classifier = classifier.train()
        pred, _ = classifier(points)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %
              (epoch,
               i,
               num_batch,
               loss.item(),correct.item() / float(opt.batch_size)))

        if i % 10 == 0:
            j, data = next(enumerate(test_dataloader, 0))
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
                   i,
                   num_batch,
                   blue('test'),
                   loss.item(),
                   correct.item()/float(opt.batch_size)))

    torch.save(classifier.state_dict(),
               '%s/cls_model_%d.pth' % (opt.output_dir, epoch))
    torch.save(embeddings.state_dict(),
               '%s/cls_embeddings_%d.pth' % (opt.output_dir, epoch))
