import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import seaborn
seaborn.set(style='whitegrid',font_scale=2.0)

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=True,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')
parser.add_argument('--c', type=int, default=4,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=700,
                    help='Num of spectra dimension')               

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

set_seed(args.seed,args.cuda)

if args.c == 3: 
    x = np.loadtxt('adulteration-x.txt')
    y = np.loadtxt('adulteration-y.txt').astype('int')
    labeltxt = ['10A','1A','P']
else:
    x = np.loadtxt('storage-x.txt')
    y = np.loadtxt('storage-y.txt').astype('int')
    labeltxt = ['3M','2M','1M','0M']

def evaluate(y,pred):
    confusion = confusion_matrix(y,pred)
    acc = accuracy_score(y,pred)
    print('Accuracy=%.3f' % acc)
    plt.figure()
    seaborn.heatmap(confusion,annot=True,cbar=False,fmt='d',
        xticklabels=labeltxt,yticklabels=labeltxt,cmap='Blues')
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        self.lin1 = nn.Linear(args.d,args.hidden)
        self.q = nn.Linear(args.hidden,args.hidden)
        self.k = nn.Linear(args.hidden,args.hidden)
        self.v = nn.Linear(args.hidden,args.hidden)
        self.a = nn.Parameter(torch.rand(1))
        self.lin2 = nn.Linear(args.hidden,args.c)

    def forward(self,x):
        x = self.lin1(x)
        w = torch.mm(self.q(x),self.k(x).t())/np.sqrt(args.hidden)
        x = self.a*x + (1-self.a)*torch.mm(torch.softmax(w,dim=0),self.v(x))
        x = self.lin2(x)
        return x

def nnclassifier(xt,yt,xv):
    xt = torch.from_numpy(xt).float()
    xv = torch.from_numpy(xv).float()
    yt = torch.LongTensor(yt)
    clf = Attention()
    opt = torch.optim.Adam(clf.parameters(),lr=args.lr,weight_decay=args.wd)
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()
    
    for e in range(args.epochs):
        clf.train()
        z = clf(xt)
        loss = F.cross_entropy(z,yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e%20 == 0 and e!=0:
            print('Epoch %d | Lossp: %.4f' % (e, loss.item()))

    clf.eval()
    z = clf(xv)
    if args.cuda: z = z.cpu()
    predict = z.detach().numpy()
    predict = predict.argmax(axis=1)
    return predict

def kfold(x,y):
    kf = StratifiedKFold(n_splits=5,shuffle=True)
    predict = np.zeros(len(y))
    for i,(train,test) in enumerate(kf.split(x,y)):
        print('Fold ',i)
        xt = x[train]
        yt = y[train]
        xv = x[test]
        predict[test] = nnclassifier(xt,yt,xv)
    
    evaluate(y,predict)

kfold(x,y)