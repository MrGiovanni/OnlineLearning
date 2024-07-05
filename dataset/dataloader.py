import sys

import torch

from monai.data import DataLoader, Dataset, list_data_collate
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    Orientationd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd,
    apply_transform,
)


sys.path.append("..") 
from dataset.mysampler import ResumableDistributedSampler
from dataset.utils import LoadImageh5d_train, LoadImageh5d_test
from dataset.utils import RandZoomd_select, RandCropByPosNegLabeld_select, RandCropByLabelClassesd_select


class MyDataset(Dataset):
    """Dataset that reads videos"""
    def __init__(self,
                 data_dict,
                 num_files,
                 transforms=None):
        super().__init__(data=data_dict, transform=transforms)
        self.data_dict = data_dict
        self.num_files = num_files
        self.filelist_mmap = None
        self.transforms = transforms
    
    def _transform(self, data_i):
        return apply_transform(self.transforms, data_i) if self.transforms is not None else data_i
    
    def __getitem__(self, index):
        data = self.data_dict[index]
        data['index'] = index
        data_i = self._transform(data)
        
        return data_i

    def __len__(self):
        return self.num_files


def get_loader(args):
    
    if args.phase == 'test':
        test_transforms = Compose(
            [
                LoadImageh5d_test(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
                ToTensord(keys=["image", "label", "post_label"]),
            ]
        )

        test_img = []
        test_lbl = []
        test_post_lbl = []
        test_name = []
        for item in args.dataset_list:
            for line in open(args.data_txt_path + item +'_test.txt'):
                name = line.strip().split()[1].split('.')[0]
                test_img.append(args.data_root_path + line.strip().split()[0])
                test_lbl.append(args.label_root_path + line.strip().split()[1])
                test_post_lbl.append(args.data_root_path + name.replace('label', 'post_label_32cls') + '.h5')
                test_name.append(name)
        data_dicts_test = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                    for image, label, post_label, name in zip(test_img, test_lbl, test_post_lbl, test_name)]
        print('test len {}'.format(len(data_dicts_test)))

        test_dataset = Dataset(data=data_dicts_test, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return test_loader, test_transforms

    elif args.phase == 'train':
        train_transforms = Compose(
            [
                LoadImageh5d_train(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
                SpatialPadd(keys=["image", "label", "post_label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
                RandZoomd_select(keys=["image", "label", "post_label"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest', 'nearest']),
                RandCropByPosNegLabeld_select(
                    keys=["image", "label", "post_label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=2,
                    neg=1,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandCropByLabelClassesd_select(
                    keys=["image", "label", "post_label"],
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    ratios=[1, 1, 5],
                    num_classes=3,
                    num_samples=args.num_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandRotate90d(
                    keys=["image", "label", "post_label"],
                    prob=0.10,
                    max_k=3,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.20,
                ),
                ToTensord(keys=["image", "label", "post_label"]),
            ]
        )

        train_img = []
        train_lbl = []
        train_post_lbl = []
        train_name = []
        for item in args.dataset_list:
            for line in open(args.data_txt_path + item +'_train.txt'):
                name = line.strip().split()[1].split('.')[0]
                train_img.append(args.data_root_path + line.strip().split()[0])
                train_lbl.append(args.label_root_path + line.strip().split()[1])
                train_post_lbl.append(args.data_root_path + name.replace('label', 'post_label_32cls') + '.h5')
                train_name.append(name)
        data_dicts_train = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                    for image, label, post_label, name in zip(train_img, train_lbl, train_post_lbl, train_name)]
        if args.local_rank == 0:
            print('train len {}'.format(len(data_dicts_train)))
        
        train_dataset = MyDataset(
            data_dicts_train,
            num_files=len(data_dicts_train),
            transforms=train_transforms)

        if args.local_rank == 0:
            print(f'Dataset: {len(train_dataset)}')
        
        train_sampler = ResumableDistributedSampler(
                dataset=train_dataset,
                shuffle=args.shuffle,
                batch_size=args.batch_size,
                drop_last=True)
        
        if args.local_rank == 0:
            print(f'Sampler: {len(train_sampler)}')

        if args.memory == 'LM':
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            memory_size = int(args.memory_size / world_size)
            from dataset.mysampler import LMBatchSampler
            batch_sampler = LMBatchSampler(
                memory_size=memory_size,
                repeat=args.sampling_rate,
                sampler=train_sampler,
                batch_size=args.batch_size,
                drop_last=True)
        elif args.memory == 'DM':
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            memory_size = int(args.memory_size/ world_size)
            from dataset.mysampler import DMBatchSampler
            batch_sampler = DMBatchSampler(
                memory_size=memory_size,
                repeat=args.sampling_rate,
                sampler=train_sampler,
                batch_size=args.batch_size,
                drop_last=True)
        elif args.memory == 'SM':
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            memory_size = int(args.memory_size/ world_size)
            from dataset.mysampler import SMBatchSampler
            batch_sampler = SMBatchSampler(
                memory_size=memory_size,
                repeat=args.sampling_rate,
                sampler=train_sampler,
                batch_size=args.batch_size,
                top_k_entropy=args.top_k_entropy,
                drop_last=True)
        else:
            raise NotImplementedError
        if args.local_rank == 0:
            print(f'Batch Sampler: {len(batch_sampler)}')
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=list_data_collate,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=1)
            
        if args.local_rank == 0:
            print(f'Train loader: {len(train_loader)}')
        
        return train_loader, batch_sampler

