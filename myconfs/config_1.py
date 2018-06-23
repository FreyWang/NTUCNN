import os
import pickle
MODEL_FILE_NAME = 'HCN'
MODEL_NAME = 'HCN'
# two layer upon the path of this file
CONFIG_NAME = os.path.basename(__file__)
PROJ_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJ_ROOT_DIR, 'data')
IMG_DIR = os.path.join(DATA_DIR, 'data_32frame')
# the path of pretrained resnet50
PRETRAIN_PATH = os.path.join(DATA_DIR, 'resnet50.pth')
MODEL_DIR = os.path.join(PROJ_ROOT_DIR, 'model_dir', 'model_{}'.format(
    os.path.basename(__file__).rsplit('.')[0].split('_')[1]))
# basename(__file__) == this filename ,rsplit means split from the right
STATE_DIR = os.path.join(MODEL_DIR, 'states')
# image data
# data name and label
IMDB_PATH = os.path.join(DATA_DIR, 'imdb.pickle')
STATE_PREFIX = 'epoch'
NUM_CLASSES = 60
BATCH_SIZE = {'train': 64, 'test': 256}
MAX_EPOCHS = 30
GPUS = [2]
DEFAULT_GPU = 2
MAX_MAP = []  # store test map when training
MAX_PREC = [] # store test precision when training
PARAM_GROUPS = [{
    'params': ['default'],
    'lr': 1e-3,
    'weight_decay': 5e-4
}, {
    'params': ['fc'],
    'lr': 1e-2,
    'weight_decay': 5e-4
}]
PATCH_NAME = 'human'

'''uncomment these lines to send logs to your email'''
# EMAIL = True
# EMAIL_ADDR = 'hyesunzhu@outlook.com'

if not os.path.exists(STATE_DIR):
    os.makedirs(STATE_DIR)


'''the inner structure in the log data'''
if not os.path.exists(os.path.join(MODEL_DIR, 'log.pickle')):
    log = {
        'train': {
            'loss': [],
            'prec': [],
            'map': []
        },
        'test': {
            'loss': [],
            'prec': [],
            'map': []
        }
    }
    with open(os.path.join(MODEL_DIR, 'log.pickle'), 'wb') as f:
        pickle.dump(log, f)

if not os.path.exists(os.path.join(MODEL_DIR, 'train_log.pickle')):
    train_log = []
    with open(os.path.join(MODEL_DIR, 'train_log.pickle'), 'wb') as f:
        pickle.dump(train_log, f)
