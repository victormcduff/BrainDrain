import argparse, os
import numpy as np
import torch
from multiprocessing import Process
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import visdom

layer_size = 1024

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
scatter_x = []
scatter_y = []

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(13521, layer_size)#5917, layer_size)
        self.hidden2 = nn.Linear(layer_size, layer_size)
        self.hidden3 = nn.Linear(layer_size, layer_size)
        self.hidden4 = nn.Linear(layer_size, layer_size)
        self.hidden5 = nn.Linear(layer_size, layer_size)
        self.hidden6 = nn.Linear(layer_size, layer_size)
        self.hidden7 = nn.Linear(layer_size, layer_size)
        self.hidden8 = nn.Linear(layer_size, 6400)
        #self.act2 = nn.ReLU()
        #self.hidden3 = nn.Linear(layer_size, 6400)
        #self.act_output = nn.ReLU() #nn.Sigmoid()

    def forward(self, x):
        #x = self.act1(self.hidden1(x))
        #x = self.act2(self.hidden2(x))
        #x = self.act_output(self.hidden2(x))

        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.hidden7(x)
        x = self.hidden8(x)
        return x


def plot_graph():
    plt.scatter(scatter_x, scatter_y)
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        type=str,
        default='',
        help="Target variable",
    )
    parser.add_argument(
        "--roi",
        required=True,
        type=str,
        nargs="*",
        help="use roi name",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs",
    )

    opt = parser.parse_args()
    target = opt.target
    subject = opt.subject
    roi = opt.roi
    epochs = opt.epochs

    mridir = f'../../mrifeat/{subject}/'
    featdir = '../../nsdfeat/subjfeat/'
    savedir = f'../../decoded/{subject}/'
    os.makedirs(savedir, exist_ok=True)

    X = []
    X_te = []
    for croi in roi:
        if 'conv' in target: # We use averaged features for GAN due to large number of dimension of features
            cX = np.load(f'{mridir}/{subject}_{croi}_betas_ave_tr.npy').astype("float32")
        else:
            cX = np.load(f'{mridir}/{subject}_{croi}_betas_tr.npy').astype("float32")
        cX_te = np.load(f'{mridir}/{subject}_{croi}_betas_ave_te.npy').astype("float32")
        X.append(cX)
        X_te.append(cX_te)

    cX = []
    cX_te = []

    X = np.hstack(X)
    X_te = np.hstack(X_te)

    Y = np.load(f'{featdir}/{subject}_each_{target}_tr.npy').astype("float32")#.reshape([X.shape[0],-1])
    Y_te = np.load(f'{featdir}/{subject}_ave_{target}_te.npy').astype("float32")#.reshape([X.shape[0],-1])

    print('shape of X, our betas to come: ', X.shape) 
    print('shape of Y, the image array: ', Y.shape)


    X = torch.tensor(X, dtype=torch.float32).to("cuda") #dtype? uni8?
    X_te = torch.tensor(X_te, dtype=torch.float32).to("cuda")
    Y = torch.tensor(Y, dtype=torch.float32).to("cuda")
    Y_te = torch.tensor(Y_te, dtype=torch.float32).to("cuda")


    model = Network().to("cuda")
    print('Model: ', model)

    batch_size = 100

    loss_function = nn.MSELoss() #mean squared loss
    optimiser = optim.Adam(model.parameters(), lr=0.00001)

    #vis = visdom.Visdom()

    #p = Process(target=plot_graph)

    #plt.scatter(scatter_x, scatter_y)
    #plt.draw()


    print('Start training with ', epochs, ' epochs')
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i+batch_size]
            Y_pred = model(Xbatch)
            Ybatch = Y[i:i+batch_size]
            loss = loss_function(Y_pred, Ybatch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            #print(f'Finished batch {i}, loss {loss}')

        scatter_x.append(epoch)
        scatter_y.append(loss.cpu().detach().numpy())

        #p.join()
        #vis.line(np.array([scatter_x, scatter_y]))
        #plt.draw()

        #ani = animation.FuncAnimation(fig, animate, interval=1)
        #plt.show(block=False)
        
        
        print(f'Finished epoch {epoch}, latest loss {loss}')

    print('Saving weights')
    torch.save(model.state_dict(), 'test2.pth')

    print('Now making a prediction...')
    prediction = model(X_te)

    accuracy = loss_function(prediction, Y_te)
    print(f'Model accuracy: {accuracy}')

    #print(prediction)

    num_pred = prediction.cpu().detach().numpy()

    print('Prediction for Image 0: ', num_pred[0])
    print('True Image 0: ', Y_te[0])
    #print(num_pred.shape)

    print('Saving the scores')
    np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}_torch.npy', num_pred)

    plt.scatter(scatter_x, scatter_y)  
    plt.show()


if __name__ == "__main__":
    main()