from pathlib import Path
import torch
from tqdm import tqdm
import sys
from data_objects.DeepSpeakerDataset import DeepSpeakerDataset

train_dataset = DeepSpeakerDataset(
    Path('/mnt/lustre/sjtu/home/czy97/sid/AutoSpeech/VoxCeleb1'), 300, 'train')
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=48,
    num_workers=sys.argv[1],
    pin_memory=True,
    shuffle=True,
    drop_last=True,
)

for data, label in tqdm(train_loader):
    pass