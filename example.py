import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import time
from utils import * 
from conformal import ConformalModel
import torch.backends.cudnn as cudnn
import random
import mmpretrain
import tqdm
import json
import matplotlib.pyplot as plt
from uncertainties import unumpy

parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet')
parser.add_argument('--data_calib', metavar='IMAGENETVALDIR', help='path to Imagenet Val')
parser.add_argument('--data_test', metavar='IMAGENETVALDIR', help='path to Imagenet Val')
parser.add_argument('--batch_size', metavar='BSZ', help='batch size', type=int, default=128)
parser.add_argument('--num_workers', metavar='NW', help='number of workers', default=0)
parser.add_argument('--num_calib', metavar='NCALIB', help='number of calibration points', default=10000)
parser.add_argument('--seed', metavar='SEED', help='random seed', default=0)
parser.add_argument('--post-process', action='store_true', help='Make summary plots without processing videos again.')

def compute_accuracy(argmax, target):
    return (argmax==target).sum() / len(target)

if __name__ == "__main__":
    args = parser.parse_args()
    ### Fix randomness 
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    
    if not args.post_process:
        # Transform as in https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L92 
        transform = transforms.Compose([
                        transforms.Resize(400),
                        transforms.CenterCrop(380),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std= [0.5, 0.5, 0.5])
                    ])

        # Get the conformal calibration dataset
        imagenet_calib_data = torchvision.datasets.ImageFolder(args.data_calib, transform)
        imagenet_val_data = torchvision.datasets.ImageFolder(args.data_test, transform)

        # Initialize loaders 
        calib_loader = torch.utils.data.DataLoader(imagenet_calib_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(imagenet_val_data, batch_size=1, shuffle=True, pin_memory=True)

        cudnn.benchmark = True

        # Get the model 
        model = mmpretrain.get_model('../mmcls/configs/cracks_efficientnet.py', pretrained='../mmcls/output/cracks_efficientnet_20230529/epoch_2.pth', device='cuda:0')
    #    model = torchvision.models.resnet152(pretrained=True,progress=True).cuda()
    #    model = torch.nn.DataParallel(model) 
        model.eval()

        # optimize for 'size' or 'adaptiveness'
        lamda_criterion = 'size'
        # allow sets of size zero
        allow_zero_sets = False 
        # use the randomized version of conformal
        randomized = True 

        # Conformalize model
        model = ConformalModel(model, calib_loader, alpha=0.1, lamda=0, randomized=randomized, allow_zero_sets=allow_zero_sets)

    #    print("Model calibrated and conformalized! Now evaluate over remaining data.")
    #    validate(val_loader, model, print_bool=True)
    #
        #print("Complete!")


        size_list = []
        target_list = []
        ranking_list = [] # ranking of true class based on predicted scores
        argmax_list = []
        covered_list = []
        for i, (x, target) in tqdm.tqdm(enumerate(val_loader)):
            target = target.cuda()
            # compute output
            output, S = model(x.cuda())
            size_list.append(len(S[0]))
            target_list.append(target[0].item())
            ranking_list.append(torch.where(output[0].sort(descending=True)[1]==target[0].item())[0].item())
            argmax_list.append(output.argmax().item())
            covered_list.append(target[0].item() in S[0])

        results = {
            'size' : size_list,
            'target' : target_list,
            'ranking' : ranking_list,
            'argmax' : argmax_list,
            'covered' : covered_list
            }
        with open('results_cracks.json', 'w') as f:
            json.dump(results, f)

    # post-processing
    with open('results_cracks.json', 'r') as f:
        results = json.load(f)

    size, target, ranking, argmax, covered = np.array(results['size']), np.array(results['target']), np.array(results['ranking']), np.array(results['argmax']), np.array(results['covered'])

    classes = [
        'Arrachement_pelade',
        'Bouche_a_clef',
        'Comblage_de_trou_ou_Projection_d_enrobe',
        'Faiencage',
        'Grille_avaloir',
        'Longitudinale',
        'Nid_de_poule',
        'Pontage_de_fissures',
        'Raccord_de_chaussee',
        'Regard_tampon',
        'Remblaiement_de_tranchees',
        'Transversale'
    ]

    print(f'{"class" : <60}{"accuracy": ^10}{"coverage": ^10}{"average size": ^15}')
    for icls, cls in enumerate(classes):
        mask = (target == icls)
        accuracy = compute_accuracy(argmax[mask], target[mask])
        coverage = covered[mask].sum() / len(covered[mask])
        avg_size = size[mask].mean()
        print(f'{cls: <60}{accuracy: ^10.2f}{coverage: ^10.2f}{avg_size: ^15.2f}')
        
    print('')
    print(f'{"prediction-set size" : <60}{"N_sample": ^10}{"coverage": ^10}')
    for isize in range(size.max()):
        mask = (size == isize+1)
        coverage = covered[mask].sum() / len(covered[mask])
        print(f'{isize+1: <60}{mask.sum(): ^10.2f}{coverage: ^10.2f}')


    print('')
    print(f'{"true-class ranking" : <60}{"N_sample": ^10}{"average size": ^15}')
    median = []
    mean = []
    uncertainty_mean = [] # statistical uncertainty on the mean size of each ranking
    for idiff in range(ranking.max()+1):
        mask = (ranking == idiff)

        median.append(np.median(size[mask]))
        
        size_hist = np.array([(size[mask]==isize+1).sum() for isize in range(size[mask].max())])
        unc_var = unumpy.uarray(size_hist.tolist(), np.sqrt(size_hist).tolist())
        unc_var_mean = (unc_var * np.arange(1,size[mask].max()+1)).sum() / unc_var.sum()
        avg_size = unumpy.nominal_values(unc_var_mean)
        uncert = unumpy.std_devs(unc_var_mean)
        mean.append(avg_size)
        uncertainty_mean.append(uncert)
        
        print(f'{idiff: <60}{mask.sum(): ^10.2f}{avg_size: ^15.2f}')


    fig, ax = plt.subplots()
    ax.set_xlabel('true-class ranking')
    ax.set_ylabel('prediction-set size')

    ax.errorbar(range(ranking.max()+1), mean, yerr=uncertainty_mean, linestyle="None", marker='o', color='black', label='mean')
    ax.plot(range(ranking.max()+1), median, linestyle="None", marker='o', color='red', label='median')
    
    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig(f'size_vs_ranking.png')
