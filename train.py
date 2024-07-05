import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from model.Universal_model import Universal_model
from dataset.dataloader import get_loader

from utils.loss import DiceLoss_SM, Multi_BCELoss_SM, DiceLoss, Multi_BCELoss
from utils.utils import adjust_learning_rate, calculate_remaining_time, TEMPLATE, NUM_CLASS
from utils.utils import AverageMeter, WindowAverageMeter, ProgressMeter, CheckpointManager

torch.multiprocessing.set_sharing_strategy('file_system')


def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE, epoch, writer, ckpt_manager):
    
    batch_time = WindowAverageMeter('Time', fmt=':6.3f')
    data_time = WindowAverageMeter('Data', fmt=':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lr_meter = AverageMeter('LR', ':.4e')
    buff_meters = []
    
    num_seen = AverageMeter('#Seen', ':6.3f')
    num_seen_max = AverageMeter('#Seen Max', ':6.3f')
    similarity = AverageMeter('Memory Sim', ':6.3f')
    neig_similarity = AverageMeter('Memory Neig Sim', ':6.3f')
    buff_meters = [num_seen, num_seen_max, similarity,
    neig_similarity]
    progress = ProgressMeter(len(train_loader), 
                             [batch_time, data_time, lr_meter] + buff_meters + [losses],
                             prefix="Epoch: [{}]".format(epoch),
                             tbwriter=writer,
                             rank=args.local_rank)
    
    model.train()
    
    end = time.time()
    start_time = time.time()
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                
    for data in train_loader:
        batch_i = train_loader.batch_sampler.advance_batches_seen()
        effective_epoch = epoch + (batch_i / len(train_loader))
        lr = adjust_learning_rate(optimizer,
                                  effective_epoch,
                                  args,
                                  epoch_size=len(train_loader))
        lr_meter.update(lr)
        
        x, y, ratio, name = data['image'], data["post_label"], data['organs_ratio'], data['name']
        data_time.update(time.time() - end)
        
        x, y, ratio = x.to(args.device), y.float().to(args.device), ratio.float().to(args.device)
        logit_map, z = model(x)
        
        term_seg_Dice = loss_seg_DICE.forward(logit_map, y, name, ratio, TEMPLATE)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y, name, ratio, TEMPLATE)
        loss_per_sample = term_seg_BCE + term_seg_Dice
        loss = loss_per_sample.mean()
        losses.update(loss.item(), x.size(0))
        
        with torch.no_grad():
            data['feature'] = z.squeeze(dim=-1).squeeze(dim=-1).squeeze(dim=-1).detach()
            data['entropy'] = term_seg_BCE.detach()
            data['loss'] = loss_per_sample.detach()
            
            stats = train_loader.batch_sampler.update_sample_stats(data)
            if 'num_seen' in stats:
                num_seen.update(stats['num_seen'].float().mean().item(),
                                stats['num_seen'].shape[0])
                num_seen_max.update(stats['num_seen'].float().max().item(),
                                    stats['num_seen'].shape[0])
            if 'similarity' in stats:
                similarity.update(stats['similarity'].float().mean().item(),
                                stats['similarity'].shape[0])
            if 'neighbor_similarity' in stats:
                neig_similarity.update(
                    stats['neighbor_similarity'].float().mean().item(),
                    stats['neighbor_similarity'].shape[0])
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Create checkpoints
        if ckpt_manager is not None:
            ckpt_manager.checkpoint(epoch=epoch,
                                    batch_i=batch_i,
                                    save_dict={
                                        'epoch': epoch,
                                        'batch_i': batch_i,
                                        'arch': args.backbone,
                                    })

        # measure elapsed time
        batch_time.update(time.time() - end)
        # measure eta time
        if batch_i % args.print_freq == 0 and args.local_rank == 0:
            days, hours, minutes, seconds = calculate_remaining_time(start_time, batch_i, len(train_loader))
            print(f"ETA: {days} DAY {hours} HR {minutes} MIN {seconds} SEC")
        
        end = time.time()

        # Log
        if batch_i % args.print_freq == 0:
            tb_step = (
                epoch * len(train_loader.dataset) // args.batch_size +
                batch_i * world_size)
            progress.display(batch_i)
            progress.tbwrite(tb_step)




def process(args):
    rank = 0

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 3D model
    model = Universal_model(out_channels=NUM_CLASS)

    #Load pre-trained weights
    if args.pretrain is not None:
        model.load_params(torch.load(args.pretrain)["state_dict"])
        if rank == 0:
            print('load pretrain')

    word_embedding = torch.load(args.word_embedding)
    model.organ_embedding.data = word_embedding.float()
    if rank == 0:
        print('load word embedding')

    model.to(args.device)
    
    
    model = DistributedDataParallel(model, device_ids=[args.device])

    # criterion and optimizer
    if args.loss_type == 'SM':
        loss_seg_DICE = DiceLoss_SM(num_classes=NUM_CLASS).to(args.device)
        loss_seg_CE = Multi_BCELoss_SM(num_classes=NUM_CLASS).to(args.device)
    else:
        loss_seg_DICE = DiceLoss(num_classes=NUM_CLASS).to(args.device)
        loss_seg_CE = Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=args.save_dir+'/' + args.log_name)
        print('Writing Tensorboard logs to ', args.save_dir+'/' + args.log_name)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader(args)

    modules = {
        'state_dict': model,
        'optimizer': optimizer,
        'sampler': train_loader.batch_sampler
    }
    ckpt_manager = CheckpointManager(
        modules=modules,
        ckpt_dir=os.path.join(args.save_dir, args.log_name),
        epoch_size=len(train_loader),
        epochs=args.max_epoch,
        save_freq=args.store_num,
        save_freq_mints=args.store_num_mints)
    if args.resume:
        args.start_epoch = ckpt_manager.resume()
    
    for epoch in range(args.start_epoch, args.max_epoch):
        
        dist.barrier()
        train_sampler.set_epoch(epoch)

        train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE, epoch, writer, ckpt_manager)
        
        ckpt_manager.checkpoint(epoch=epoch + 1,
                                save_dict={
                                    'epoch': epoch + 1,
                                    'batch_i': 0,
                                    'arch': args.backbone,
                                })
        train_sampler.init_from_ckpt = False
        
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ## Distributed training
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    
    ## Logging
    parser.add_argument('--print_freq', default=10, help='The path resume from checkpoint')
    parser.add_argument('--log_name', default='{}-{}-x{}-mem{}-loss{}', help='The path resume from checkpoint')
    parser.add_argument('--save_dir', default='./out/{}', help='The path resume from checkpoint')

    ## Model
    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--backbone', default='unet')
    parser.add_argument('--resume', default=False, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='pretrained_weights/Genesis_Chest_CT.pt')
    parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding.pth', 
                        help='The path of word embedding')
    
    ## Hyperparameter
    parser.add_argument("--start_epoch", default=0)
    parser.add_argument('--max_epoch', default=1, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=50, type=int, help='Store model how often')
    parser.add_argument('--store_num_mints', default=30, type=int, help='Store model how often (minutes)')
    
    ## Optimizer
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--lr_schedule', default='cos', help='memmap: constant, other: cos', choices=['constant', 'cos', 'triangle'])
    parser.add_argument('--lr_schedule_period', default=3000, help='Learning rate')
    parser.add_argument('--max_lr', default=0.003, type=float, help='Learning rate')
    parser.add_argument('--exit_decay', default=0.0, type=float, help='Learning rate')
    
    ## Memory
    parser.add_argument('--loss_type', default='SM', choices=['Simple', 'SM'])
    parser.add_argument('--memory', default='SM', choices=['LM', 'DM', 'SM'])
    parser.add_argument('--sampling_rate', default=100)
    parser.add_argument('--memory_size', default=128)
    parser.add_argument('--top_k_entropy', default=4, help='memory()/k')
    parser.add_argument('--shuffle', default=False)
    
    ## Dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT'])
    parser.add_argument('--data_root_path', default='', help='data root path')
    parser.add_argument('--label_root_path', default='', help='label root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, help='batch size')
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
    args.log_name = args.log_name.format(args.backbone, args.memory, args.sampling_rate, args.buff_siz, args.loss_type)
    args.save_dir = args.save_dir.format(args.dataset_list[0])

    os.makedirs(os.path.join(args.save_dir, args.log_name), exist_ok=True)
    message = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    with(open(os.path.join(args.save_dir, args.log_name, 'args.txt'), 'w')) as f:
        f.write(message)
    
    process(args=args)

if __name__ == "__main__":
    main()