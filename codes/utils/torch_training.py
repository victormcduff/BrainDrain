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
from nsd_access.nsda import NSDAccess
from tqdm import tqdm
import scipy.io
import h5py
from PIL import Image

from networks import *

path_to_disk = '/data/ArkadiyArchive/Brain/'
path_to_data = path_to_disk + 'NSA'

input_layer = 6400#9600#5917#10500 #9500 #8185 #7604
output_layer = 6400 #59136
layer_size = 1024

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
scatter_x = []
scatter_y = []

def normalise_1d_layers(layer, desired_size):
    print('Attempting to normalise array with shape', layer.shape)

    thick_of_it = len(layer[:,0])
    layer_norm = np.zeros((thick_of_it,desired_size)) #np.zeros((thick_of_it,desired_size))

    #for i in range(thick_of_it):
     #   for j in range(len(layer[i])):
      #      layer_norm[i,j] = layer[i,j]

    for j in range(len(layer[0])):
        layer_norm[:,j] = layer[:,j]

    return layer_norm#.reshape(thick_of_it, 1, 40*2, 40*2)


def get_test_train_images(subject):
    def scan_through_stim(stims):
        imgs = np.zeros((len(stims), 3, 425, 425))#[]
        tr_idx = np.zeros(len(stims))
        #imgs_tr = np.zeros((len(stims), 3, 425, 425))
        #imgs_te = np.zeros((len(stims), 3, 425, 425))

        for idx, s in tqdm(enumerate(stims)): 
            if s in sharedix:
                tr_idx[idx] = 0
            else:
                tr_idx[idx] = 1

            img = sdataset[idx,:,:,:].reshape(3,425,425) / 255.0
            #imgs.append(img)
            imgs[idx] = img.astype("float16")

        #imgs = np.stack(imgs).astype("float32")

        imgs_tr = imgs[tr_idx==1,:]
        imgs_te = imgs[tr_idx==0,:]

        imgs = []

        return imgs_tr, imgs_te

    nsda = NSDAccess(path_to_data)
    sf = h5py.File(nsda.stimuli_file, 'r')
    sdataset = sf.get('imgBrick')

    nsd_expdesign = scipy.io.loadmat('../../nsd/nsd_expdesign.mat')
    sharedix = nsd_expdesign['sharedix']-1 

    stims_ave = np.load(f'{path_to_disk}/mrifeat/{subject}/{subject}_stims_ave.npy')
    stims_each = np.load(f'{path_to_disk}/mrifeat/{subject}/{subject}_stims.npy')
    
    Y_ave_tr, Y_ave_te = scan_through_stim(stims_ave)
    Y_each_tr, Y_each_te = scan_through_stim(stims_each)

    Y_ave_tr = []
    Y_each_te = []

    #print('Y ave te shape: ', Y_ave_te.shape)
    return Y_each_tr, Y_ave_te    


def get_img_index(stims, sharedindx, train):
    tr_indx = []
    for idx, s in tqdm(enumerate(stims)): 
        if s in sharedindx: #test
            if not train:
                tr_indx.append(idx)
        else: #train
            if train:
                tr_indx.append(idx)
    
    return tr_indx


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
        type=int,
        default=1,
        nargs="*",
        required=True,
        help="subject id, 1-8 (1,2, 5, 7 have full data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs",
    )

    opt = parser.parse_args()
    target = opt.target
    subject_id = opt.subject
    roi = opt.roi
    epochs = opt.epochs

    featdir = f'{path_to_disk}/nsdfeat/subjfeat/'
    weights_directory = f"{path_to_disk}/network_models/"
    subject = ''

    if len(subject_id) == 1:
        subject = 'subj0' + str(subject_id[0])
        
    elif len(subject_id) > 1:
        subject = 'subjs' + str(min(subject_id)) + "-" + str(max(subject_id))

    savedir = f'{path_to_disk}/decoded/{subject}/'
    weights_file = weights_directory + f"/{subject}_{roi}_{target}_weights_vol.pth"
    prediction_file = f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}_torch_vol.npy'

    os.makedirs(weights_directory, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)

    X = []
    X_te = []
    Y = []
    Y_te = []

    for i in subject_id:
        print('Doing subject', i)
        subject = 'subj0' + str(i)

        mridir = f'{path_to_disk}/mrifeat/{subject}/'
    
        roiX = []
        roiX_te = []
        for croi in roi:
            print('ROI: ', croi)
            if 'conv' in target: # We use averaged features for GAN due to large number of dimension of features
                cX = np.load(f'{mridir}/{subject}_{croi}_tensorbetas_ave_tr.npy').astype("float32")
            else:
                cX = np.load(f'{mridir}/{subject}_{croi}_tensorbetas_tr.npy').astype("float32")
            cX_te = np.load(f'{mridir}/{subject}_{croi}_tensorbetas_ave_te.npy').astype("float32")

            print('cX shape for this roi: ', cX.shape)

            roiX.append(cX)
            roiX_te.append(cX_te) #need only one not all subj

        X.append(np.hstack(roiX)) #normalise_1d_layers(np.hstack(roiX), input_layer)) #being it to the same size (network input layer)
        X_te.append(np.hstack(roiX_te))#normalise_1d_layers(np.hstack(roiX_te), input_layer))

        cY = np.load(f'{featdir}/{subject}_each_{target}_tr.npy').astype("float32")#.reshape([X.shape[0],-1])
        cY_te = np.load(f'{featdir}/{subject}_ave_{target}_te.npy').astype("float32")#.reshape([X.shape[0],-1])

        cY = cY.reshape(len(cY),4,40,40)#.unsqueeze(0)
        cY_te = cY_te.reshape(len(cY_te),4,40,40)#.unsqueeze(0)
        #cY, cY_te = get_test_train_images(subject)

        Y.append(cY)
        Y_te.append(cY_te)

        cX = []
        cX_te = []
        roiX = []
        roiX_te = []
        cY = []
        cY_te = []

    #Y = np.hstack(Y)
    #Y_te = np.hstack(Y_te)

    X = np.vstack(X)
    X_te = np.vstack(X_te)
    Y = np.vstack(Y)
    Y_te = np.vstack(Y_te)

    print('shape of X, our betas to come: ', X.shape) 
    print('shape of Y, the image array: ', Y.shape)

    X = torch.tensor(X, dtype=torch.float32).to("cuda") #dtype? uni8?
    X_te = torch.tensor(X_te, dtype=torch.float32).to("cuda")
    Y = torch.tensor(Y, dtype=torch.float32).to("cuda")
    Y_te = torch.tensor(Y_te, dtype=torch.float32).to("cuda")

    model = NetworkVolume().to("cuda")
    #decoder_layer = nn.TransformerDecoderLayer(d_model=input_layer, nhead=8)
    #model = nn.TransformerDecoder(decoder_layer, num_layers=8).to("cuda")
    print('Model: ', model)

    loss_function = nn.MSELoss() #mean squared loss

    train = True

    """
    nsda = NSDAccess(path_to_data)
    sf = h5py.File(nsda.stimuli_file, 'r')
    sdataset = sf.get('imgBrick')

    nsd_expdesign = scipy.io.loadmat('../../nsd/nsd_expdesign.mat')
    sharedix = nsd_expdesign['sharedix']-1 

    stims_ave = np.load(f'{path_to_disk}/mrifeat/{subject}/{subject}_stims_ave.npy')
    stims_each = np.load(f'{path_to_disk}/mrifeat/{subject}/{subject}_stims.npy')

    tr_indx = get_img_index(stims_each, sharedix, True)
    te_indx = get_img_index(stims_ave, sharedix, False)
    """

    if train:
        batch_size = 100

        optimiser = optim.Adam(model.parameters(), lr=0.00001)

        print('Start training with ', epochs, ' epochs')
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                Xbatch = X[i:i+batch_size]
                Y_pred = model(Xbatch)

                Ybatch = Y[i:i+batch_size]

                #Ybatch = sdataset[tr_indx[i:i+batch_size],:,:,:].reshape(batch_size, 3,425,425) / 255.0
                #Ybatch = torch.tensor(Ybatch, dtype=torch.float32).to("cuda")

                loss = loss_function(Y_pred, Ybatch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            scatter_x.append(epoch)
            scatter_y.append(loss.cpu().detach().numpy())
            
            print(f'Finished epoch {epoch}, latest loss {loss}')

        print('Saving weights')
        torch.save(model.state_dict(), weights_file)

    else:
        print('Loading weights...')
        model.load_state_dict(torch.load('test2.pth'))

    print('Now making a prediction...')
    prediction = model(X_te)

    #Y_test = sdataset[te_indx,:,:,:].reshape(1000, 3,425,425) / 255.0
    #Y_test = torch.tensor(Y_test, dtype=torch.float32).to("cuda")

    accuracy = loss_function(prediction, Y_te)
    print(f'Model accuracy: {accuracy}')

    num_pred = prediction.cpu().detach().numpy()

    #print('Prediction for Image 0: ', num_pred[0])
    #print('True Image 0: ', Y_te[0])
    #print(num_pred.shape)

    print('Saving the scores')
    np.save(prediction_file, num_pred)

    plt.scatter(scatter_x, scatter_y)  
    plt.show()


if __name__ == "__main__":
    main()