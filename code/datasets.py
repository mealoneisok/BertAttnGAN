import pickle
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import cfg

def prepare_data(data):
    imgs, caps, attn_masks, keys = data
    real_imgs = []
    for i in range(len(imgs)):
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))
    caps = Variable(caps)
    attn_masks = Variable(attn_masks)

    if cfg.CUDA:
        caps = caps.cuda()
        attn_masks = attn_masks.cuda()

    return real_imgs, caps, attn_masks, keys


class TextDataset(Dataset):
    def __init__(self, data_dir, split='train', base_size = cfg.TREE.BASE_SIZE, transform=None):
        self.split = split
        self.data_dir = data_dir
        self.transform = transform
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.imsize = []
        for i in range(3):
            self.imsize.append(base_size)
            base_size = base_size * 2

        if split == 'train':
            with open(data_dir + '/filename_train.pkl', 'rb') as f:
                self.filenames = pickle.load(f)
            with open(data_dir + '/token_bert_train.pkl', 'rb') as f:
                self.captions = pickle.load(f)
            with open(data_dir + '/attn_mask_train.pkl', 'rb') as f:
                self.attn_mask = pickle.load(f)
            with open(data_dir + '/caps_text_train.pkl', 'rb') as f:
                self.text = pickle.load(f)
        else:
            with open(data_dir + '/filename_test.pkl', 'rb') as f:
                self.filenames = pickle.load(f)
            with open(data_dir + '/token_bert_test.pkl', 'rb') as f:
                self.captions = pickle.load(f)
            with open(data_dir + '/attn_mask_test.pkl', 'rb') as f:
                self.attn_mask = pickle.load(f)
            with open(data_dir + '/caps_text_test.pkl', 'rb') as f:
                self.text = pickle.load(f)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # get the image
        img_path = '{}/images/{}.jpg'.format(self.data_dir, filename)
        img = self.get_img(img_path)

        # choose a caption for this image
        cap_idx = np.random.randint(0, len(self.captions[idx]))
        cap = torch.tensor(self.captions[idx][cap_idx])
        attn_mask = torch.tensor(self.attn_mask[idx][cap_idx])
        key = (idx, cap_idx)
        return img, cap, attn_mask, key

    def __len__(self):
        return len(self.filenames)

    def get_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        ret = []
        for i in range(3):
            if i < 2:
                re_img = transforms.Resize(self.imsize[i])(img)
            else:
                re_img = img
            ret.append(self.normalize(re_img))
        return ret
