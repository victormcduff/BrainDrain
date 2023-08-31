import argparse, os
import numpy as np
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from himalaya.progress_bar import ProgressBar
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

def check_partial_latents(array_x, array_y, target):
    partial_image = np.zeros(len(np.load(f'../../nsdfeat/{target}/000000.npy')))

    x_partial = []
    y_partial = []

    for i in range(int(len(array_y)/3)): #a bit of cheating
        image = array_y[i]
        if image.all() != partial_image.all():
            x_partial.append(array_x[i])
            y_partial.append(image)

    return np.asarray(x_partial), np.asarray(y_partial)

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

    opt = parser.parse_args()
    target = opt.target
    roi = opt.roi

    #torch cuda for GPU; I'd be insane to ran this on CPU
    backend = set_backend("numpy", on_error="warn")#"torch_cuda") 
    subject=opt.subject

    if target == 'c' or target == 'init_latent': # CVPR
        alpha = [0.000001, 0.00001,0.0001,0.001,0.01, 0.1, 1]
    else: # text / GAN / depth decoding (with much larger number of voxels)
        alpha = [10000, 20000, 40000]

    ridge = RidgeCV(alphas=alpha)

    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
    )

    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )   

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

    """
    X, Y = check_partial_latents(X, Y, target)
    X_te, Y_te = check_partial_latents(X_te, Y_te, target)

    print('shape of X after rooting out incomplete latents: ', X.shape) 
    print('shape of new Y: ', Y.shape)
    """
    
    print(f'Now making decoding model for... {subject}:  {roi}, {target}')
    print(f'X {X.shape}, Y {Y.shape}, X_te {X_te.shape}, Y_te {Y_te.shape}')
    
    pipeline.fit(X, Y)

    print('Now making a prediction...')
    scores = pipeline.predict(X_te)

    print('Save the scores')
    np.save(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}.npy',scores)

    scores_np = np.load(f'{savedir}/{subject}_{"_".join(roi)}_scores_{target}.npy').astype("float32")
    scores_tensor = torch.tensor(scores, dtype=torch.float32).to("cuda")
    Y_te = torch.tensor(Y_te, dtype=torch.float32).to("cuda")

    loss_function = nn.MSELoss()
    accuracy = loss_function(scores_tensor, Y_te)
    print(f'Model accuracy: {accuracy}')

    #print('Evaluating correlation score')
    #rs = correlation_score(Y_te.T,scores.T)

    
    #print(f'Prediction accuracy is: {np.mean(rs, dtype=torch.dtype):3.3}')
    #print(ridge.coef_) #gets model weights?

if __name__ == "__main__":
    main()
