import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from monai.inferers import sliding_window_inference

from model.Universal_model import Universal_model
from dataset.dataloader import get_loader
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process, threshold_organ

torch.multiprocessing.set_sharing_strategy('file_system')


def test(model, ValLoader, val_transforms, args):
    save_dir = args.save_dir + '/' + args.log_name + f'/test_healthp_{args.epoch}'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir+'/predict')
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS))
    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        with torch.no_grad():
            pred, z = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        B = pred_hard.shape[0]
        for b in range(B):
            content = 'case%s| '%(name[b])
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, args.log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1],args)
            pred_hard_post = torch.tensor(pred_hard_post)

            for organ in organ_list:
                if torch.sum(label[b,organ-1,:,:,:].cuda()) != 0:
                    dice_organ, recall, precision = dice_score(pred_hard_post[b,organ-1,:,:,:].cuda(), label[b,organ-1,:,:,:].cuda())
                    dice_list[template_key][0][organ-1] += dice_organ.item()
                    dice_list[template_key][1][organ-1] += 1
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice_organ.item())
                    print('%s: dice %.4f, recall %.4f, precision %.4f.'%(ORGAN_NAME[organ-1], dice_organ.item(), recall.item(), precision.item()))
            print(content)
        
        if args.store_result:
            pred_sigmoid_store = (pred_sigmoid.cpu().numpy() * 255).astype(np.uint8)
            label_store = (label.numpy()).astype(np.uint8)
            np.savez_compressed(save_dir + '/predict/' + name[0].split('/')[0] + name[0].split('/')[-1], 
                            pred=pred_sigmoid_store, label=label_store)
            ### testing phase for this function
            one_channel_label_v1, one_channel_label_v2 = merge_label(pred_hard_post, name)
            batch['one_channel_label_v1'] = one_channel_label_v1.cpu()
            batch['one_channel_label_v2'] = one_channel_label_v2.cpu()

            _, split_label = merge_label(batch["post_label"], name)
            batch['split_label'] = split_label.cpu()
            
            visualize_label(batch, save_dir + '/output/' + name[0].split('/')[0] , val_transforms)
            
            
        torch.cuda.empty_cache()
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))

    with open(args.save_dir+'/'+args.log_name+f'/test_{args.epoch}.txt', 'w') as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content = 'Task%s| '%(key)
            for organ in organ_list:
                
                dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
            print(content)
            f.write(content)
            f.write('\n')
        content = 'Average | '
        for i in range(NUM_CLASS):
            content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])
        print(content)
        f.write(content)    
        f.write('\n')
        print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
        f.write('%s: %.4f, '%('average', np.mean(ave_organ_dice[0] / ave_organ_dice[1])))
        f.write('\n')



def main():
    parser = argparse.ArgumentParser()
    ## Distributed training
    parser.add_argument("--epoch", default=0)
    
    ## Logging
    parser.add_argument('--log_name', default=None, help='The path resume from checkpoint')
    parser.add_argument('--save_dir', default='./out/{}', help='The path resume from checkpoint')
    
    ## Model
    parser.add_argument('--resume', default='out/PAOT/PATH_TO_CHECKPOINT', help='The path resume from checkpoint')
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet]')
    
    ## Hyperparameters
    parser.add_argument('--phase', default='test', help='train or test')
    parser.add_argument('--store_result', action="store_true", default=True, help='whether save prediction result')
    
    ## Dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT'], choices=['PAOT', 'felix'])
    parser.add_argument('--data_root_path', default='', help='data root path')
    parser.add_argument('--label_root_path', default='', help='label root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    args = parser.parse_args()
    args.log_name = args.resume.split('/')[2]
    args.save_dir = args.save_dir.format(args.dataset_list[0])
    args.epoch = args.resume.split('/')[-1].split('.')[0]
    
    # prepare the 3D model
    model = Universal_model(out_channels=NUM_CLASS)
    
    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.resume)
    load_dict = checkpoint['state_dict']
    
    for key, value in load_dict.items():
        if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
            name = '.'.join(key.split('.')[1:])
            name = 'backbone.' + name
        else:
            name = '.'.join(key.split('.')[1:])
        store_dict[name] = value


    model.load_state_dict(store_dict)
    print('Use pretrained weights')
    model.cuda()

    torch.backends.cudnn.benchmark = True

    test_loader, test_transforms = get_loader(args)

    test(model, test_loader, test_transforms, args)

if __name__ == "__main__":
    main()

