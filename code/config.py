from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset name: birds
__C.DATASET_NAME = 'birds'
__C.CONFIG_NAME = ''
__C.DATA_DIR = './bird'
__C.BERT_PATH = 'uncased_L-4_H-256_A-4'
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 6

__C.B_VALIDATION = False

__C.TREE = edict()
__C.TREE.BASE_SIZE = 32

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.MAX_EPOCH = 350
__C.TRAIN.SNAPSHOT_INTERVAL = 100
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''
__C.TRAIN.B_NET_D = True

# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.Z_DIM = 100
__C.GAN.CONDITION_DIM = 100
__C.GAN.R_NUM = 2

__C.TEXT = edict()
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 25
