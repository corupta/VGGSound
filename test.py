import os
import pathlib
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import argparse
import csv
from model import AVENet
from datasets import GetAudioVideoDataset




def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default='/scratch/shared/beegfs/hchen/train_data/VGGSound_final/audio/',
        type=str,
        help='Directory path of data')
    parser.add_argument(
        '--result_path',
        default='/scratch/shared/beegfs/hchen/prediction/audioclassification/vggsound/resnet18/',
        type=str,
        help='Directory path of results')
    parser.add_argument(
        '--summaries',
        default='/scratch/shared/beegfs/hchen/epoch/audioclassification_f/resnet18_vlad/model.pth.tar',
        type=str,
        help='Directory path of pretrained model')
    parser.add_argument(
        '--pool',
        default="vlad",
        type=str,
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--csv_path',
        default='./data/',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--test',
        default='test.csv',
        type=str,
        help='test csv files')
    parser.add_argument(
        '--batch_size', 
        default=32, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=309,
        type=int,
        help=
        'Number of classes')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    return parser.parse_args() 

def save_probs(f, v_id, probs):
    curr_line = "{}".format(v_id)
    for prob in probs:
        curr_line += ",{:.5f}".format(prob)
    curr_line += "\n"
    f.write(curr_line)

TOP_X = 6

def save_prob_summary(classes, f, probs):
    f.write("Top {} Audio Classes\n".format(TOP_X))
    idx = np.argsort(probs)[::-1]
    for i in range(TOP_X):
        f.write("{:.4f} => {}\n".format(probs[idx[i]], classes[idx[i]]))
    f.write("\n")

def main():
    args = get_arguments()

    # create prediction directory if not exists
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    # init network
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model= AVENet(args) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = model.cuda()
    
    # load pretrained models
    checkpoint = torch.load(
        args.summaries, 
        map_location=(None if torch.cuda.is_available() else torch.device('cpu'))
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print('load pretrained model.')

    # create dataloader
    testdataset = GetAudioVideoDataset(args,  mode='test')
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers = 4)

    softmax = nn.Softmax(dim=1)
    print("Loaded dataloader.")
    classes = testdataset.classes

    with torch.no_grad():
        model.eval()
        with open(os.path.join(args.result_path, "audio_classification_stats.csv"), "w") as f2:
            heading_line = "video_id,{}\n".format(",".join(["\"{}\"".format(cls) for cls in classes]))
            f2.write(heading_line)
            for step, (spec, audio, label, name) in enumerate(testdataloader):
                print('%d / %d' % (step,len(testdataloader) - 1))
                spec = Variable(spec)
                label = Variable(label)
                if torch.cuda.is_available():
                    spec = spec.cuda()
                    label = label.cuda()
                aud_o = model(spec.unsqueeze(1).float())

                prediction = softmax(aud_o)

                for i, item in enumerate(name):
                    audio_id = name[i][:-4]
                    audio_dir = os.path.join(args.result_path, audio_id)
                    pathlib.Path(audio_dir).mkdir(parents=True, exist_ok=True)

                    probs = prediction[i].cpu().data.numpy()
                    v_id = audio_id
                    save_probs(f2, v_id, probs)
                    np.save(os.path.join(audio_dir, "data.npy"),prediction[i].cpu().data.numpy())
                    with open(os.path.join(audio_dir, "audio_classification_stats.csv"), 'w') as f:
                        f.write(heading_line)
                        save_probs(f, v_id, probs)

                    with open(os.path.join(audio_dir, "audio_classification_summary.txt"), 'w') as f:
                        f.write("Audio Classification Summary for Video #{}\n\n".format(v_id))
                        save_prob_summary(classes, f, probs)


                    # print example, scores
                    # print(name[i][:-4], label, prediction[i].cpu().data.numpy())



if __name__ == "__main__":
    main()

