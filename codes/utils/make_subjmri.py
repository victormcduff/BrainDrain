import argparse
import os
import numpy as np
import pandas as pd
from nsd_access import NSDAccess
import scipy.io

path_to_disk = '/data/ArkadiyArchive/Brain/'
path_to_data = path_to_disk + 'NSA'
atlasname = 'streams'

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        default='subj01',
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    parser.add_argument(
        "--session_end",
        type=int,
        default=40,
        help="final session to use",
    )

    parser.add_argument(
        "--session_start",
        type=int,
        default=1,
        help="first session to use",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=5,
        help="number of sessions to load into RAM at once",
    )

    parser.add_argument(
        "--flat_brains",
        type=int,
        default=0,
        help="0 or 1, use flat brains or not",
    )


    opt = parser.parse_args()
    subject = opt.subject
    batch = opt.batch
    flat_brains = opt.flat_brains
    session_start = opt.session_start
    session_end = opt.session_end
    sessions = session_end - session_start

    truncated_directory = 'roi_truncated_volume_arrays'
    if flat_brains:
        truncated_directory = 'roi_truncated_arrays'
    
    nsda = NSDAccess(path_to_data)
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsd_expdesign.mat')

    savedir = f'{path_to_disk}/mrifeat/{subject}/'
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(f"{savedir}/{truncated_directory}/", exist_ok=True)

    # Note that most of nsd_expdesign indices are 1-base index!
    # This is why subtracting 1
    sharedix = nsd_expdesign['sharedix']-1
    
    print('making BEHAVIOUR')
    behs = pd.DataFrame()
    for i in range(1,session_end+1):
        beh = nsda.read_behavior(subject=subject, 
                                session_index=i)
        behs = pd.concat((behs,beh))

    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    stims_unique = behs['73KID'].unique() - 1
    stims_all = behs['73KID'] - 1

    #if not os.path.exists(f'{savedir}/{subject}_stims.npy'):
    np.save(f'{savedir}/{subject}_stims.npy',stims_all)
    np.save(f'{savedir}/{subject}_stims_ave.npy',stims_unique)

    print('creation of ATLAS, the one who holds the sky')
    atlas = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')

    print('start reading BETAS')
    batches = int(sessions/batch)+2

    print('batches: ', batches-1)

    return batches, batch, stims_all, stims_unique, subject, savedir, atlas, nsda, sharedix, flat_brains, truncated_directory


def truncate_full_scans(batches, batch, subject, savedir, atlas, nsda, flat_brains, truncated_directory):
    for i in range(1, batches):
        print('beta batch', i)
        betas_all = [] #clearing memory 
        beta_trial = [] #clearing memory
        for j in range(1,batch+1):
            session_id = j+(i-1)*batch
            print('reading BETA ', session_id)
            beta_trial = nsda.read_betas(subject=subject, 
                                    session_index=session_id, 
                                    trial_index=[], # empty list as index means get all for this session
                                    data_type='betas_fithrf_GLMdenoise_RR',
                                    data_format='func1pt8mm')
            if j==1:
                betas_all = beta_trial
            else:
                betas_all = np.concatenate((betas_all,beta_trial),0)    


        print('the shape of BATCH BETAS TO COME: ', betas_all.shape)    

        for roi,val in atlas[1].items():
            print('ROI, val: ', roi,val)
            if val == 0: #Unknown in streams
                print('SKIP')
                continue
            else:
                if flat_brains:
                    betas_roi = betas_all[:,atlas[0].transpose([2,1,0])==val]
                else: #ATLAS AXIS 0 AND 2 ARE INVERTED!!!
                    b_k, b_j, b_i = np.where(atlas[0] == val) #the limits of our magic box with BRAINS
                    betas_roi = betas_all[:,min(b_i):max(b_i),min(b_j):max(b_j),min(b_k):max(b_k)] 

            print('betas ROI shape:', betas_roi.shape)

            np.save(f'{savedir}/{truncated_directory}/{subject}_{roi}_betas_batch_{i}.npy',betas_roi)


def prepare_train_data(batches, stims_all, stims_unique, subject, savedir, atlas, sharedix, flat_brains, truncated_directory):
    print('start preparing TEST/TRAIN data')
    for roi,val in atlas[1].items():
        #betas_roi = [] #memory clearing

        if val == 0: #Unknown in streams
            print('SKIP')    
            continue
        else:
        
            #load all batches for this ROI
            for i in range(1,batches):
                betas_load_batch = np.load(f'{savedir}/{truncated_directory}/{subject}_{roi}_betas_batch_{i}.npy')
                
                if i==1:
                    betas_roi = betas_load_batch
                else:
                    betas_roi = np.concatenate((betas_roi,betas_load_batch),0)

            print('full ROI beta array size: ', betas_roi.shape)

            betas_roi_ave = []
            for stim in stims_unique:
                stim_mean = np.mean(betas_roi[stims_all == stim],axis=0)
                betas_roi_ave.append(stim_mean)
            betas_roi_ave = np.stack(betas_roi_ave)
            print('beta roi average size: ', betas_roi_ave.shape)

            # Train/Test Split
            # ALLDATA
            betas_tr = []
            betas_te = []

            print('start splitting into train and test for this ROI')
            for idx,stim in enumerate(stims_all):
                if stim in sharedix:
                    betas_te.append(betas_roi[idx])
                else:
                    betas_tr.append(betas_roi[idx])

            betas_tr = np.stack(betas_tr)
            betas_te = np.stack(betas_te)

            betas_ave_tr = []
            betas_ave_te = []
            for idx,stim in enumerate(stims_unique):
                if stim in sharedix:
                    betas_ave_te.append(betas_roi_ave[idx])
                else:
                    betas_ave_tr.append(betas_roi_ave[idx])
            betas_ave_tr = np.stack(betas_ave_tr)
            betas_ave_te = np.stack(betas_ave_te)

            print('Saving train/test for this ROI')

            volume_add = 'tensor'
            if flat_brains:
                volume_add = ''

            np.save(f'{savedir}/{subject}_{roi}_{volume_add}betas_tr.npy',betas_tr)
            np.save(f'{savedir}/{subject}_{roi}_{volume_add}betas_te.npy',betas_te)
            np.save(f'{savedir}/{subject}_{roi}_{volume_add}betas_ave_tr.npy',betas_ave_tr)
            np.save(f'{savedir}/{subject}_{roi}_{volume_add}betas_ave_te.npy',betas_ave_te)


def main(): #awful awful AWFUL way of doing this whole thing, will redo it later
    batches, batch, stims_all, stims_unique, subject, savedir, atlas, nsda, sharedix, flat_brains, truncated_directory = prepare()
    truncate_full_scans(batches, batch, subject, savedir, atlas, nsda, flat_brains, truncated_directory) #comment this if already have truncated files
    prepare_train_data(batches, stims_all, stims_unique, subject, savedir, atlas, sharedix, flat_brains, truncated_directory)


if __name__ == "__main__":
    main()
