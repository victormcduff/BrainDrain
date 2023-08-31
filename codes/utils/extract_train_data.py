import h5py
from PIL import Image
import scipy.io
import argparse, os
import numpy as np

from nsd_access.nsda import NSDAccess

path_to_data = '/data/ArkadiyArchive/Brain/NSA'

nsda = NSDAccess(path_to_data)
sf = h5py.File(nsda.stimuli_file, 'r')
sdataset = sf.get('imgBrick')

stims_ave = np.load(f'../../mrifeat/subj01/subj01_stims_ave.npy')
nsd_expdesign = scipy.io.loadmat('../../nsd/nsd_expdesign.mat')
sharedix = nsd_expdesign['sharedix']-1 

os.makedirs('../../train_img', exist_ok=True)

tr_idx = np.zeros_like(stims_ave)
for idx, s in enumerate(stims_ave):
	if s in sharedix:
		tr_idx[idx] = 0
	else:
		tr_idx[idx] = 1

for i in range(1, 1001):
	print('Saving test image ', i)
	imgidx_te = np.where(tr_idx==0)[0][i] # Extract test image index
	idx73k=stims_ave[imgidx_te]
	Image.fromarray(np.squeeze(sdataset[idx73k,:,:,:]).astype(np.uint8)).save(os.path.join('../../train_img/', f"{i:05}_org.png"))  