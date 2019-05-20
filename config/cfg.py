from yacs.config import CfgNode as cfg
#config tree
Cfg = cfg()

Cfg.DATALOADER = cfg()
Cfg.DATALOADER.LOG_DIR = "./log/" #log dir and saved model dir
Cfg.DATALOADER.DATALOADER_NUM_WORKERS = 8
Cfg.DATALOADER.DATA_DIR = "/data/lujj/datasets/XXX/"
Cfg.DATALOADER.SAMPLER = 'softmax'


Cfg.MODEL = cfg()
Cfg.MODEL.INPUT_SIZE = [256, 256] #HxW
Cfg.MODEL.NAME = "resnet50"
Cfg.MODEL.DEVICE_ID = "4"#
Cfg.MODEL.LAST_STRIDE = 1
Cfg.MODEL.PRETRAIN_PATH = "/data/lujj/pretrainedmodel/resnet50-19c8e357.pth"
Cfg.MODEL.MODEL_NECK = 'bnneck'#'bnneck'
Cfg.MODEL.NECK_FEAT = "after"
Cfg.MODEL.PRETRAIN_CHOICE = 'imagenet'

Cfg.MODEL.LSOFTMAX_MARGIN = 2

Cfg.LOSS = cfg()
Cfg.LOSS.TYPE = 'softmax'
Cfg.LOSS.LABELSMOOTH = 'on'

Cfg.SOLVER = cfg()
Cfg.SOLVER.BATCHSIZE = 64
Cfg.SOLVER.OPTIMIZER = 'Adam'
Cfg.SOLVER.BASE_LR = 0.001

Cfg.SOLVER.CE_LOSS_WEIGHT = 0.25
Cfg.SOLVER.TRIPLET_LOSS_WEIGHT = 1-Cfg.SOLVER.CE_LOSS_WEIGHT
Cfg.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

Cfg.SOLVER.WEIGHT_DECAY = 0.0005
Cfg.SOLVER.BIAS_LR_FACTOR = 2
Cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0
Cfg.SOLVER.MOMENTUM = 0.9
Cfg.SOLVER.CENTER_LR = 0.5
Cfg.SOLVER.MARGIN = 0.3

Cfg.SOLVER.STEPS = [30,50,70]
Cfg.SOLVER.GAMMA = 0.1
Cfg.SOLVER.WARMUP_FACTOR = 0.01
Cfg.SOLVER.WARMUP_EPOCHS = 10
Cfg.SOLVER.WARMUP_METHOD = "linear" #option: 'linear','constant'
Cfg.SOLVER.LOG_PERIOD = 50 #iteration of display training log
Cfg.SOLVER.CHECKPOINT_PERIOD = 5 #save model period
Cfg.SOLVER.EVAL_PERIOD = 5 #validation period
Cfg.SOLVER.MAX_EPOCHS = 90

Cfg.TEST = cfg()
Cfg.TEST.IMS_PER_BATCH = 128
Cfg.TEST.FEAT_NORM = "yes"
Cfg.TEST.WEIGHT = ''
Cfg.TEST.DIST_MAT = Cfg.DATALOADER.LOG_DIR+"dist_mat.npy"
Cfg.TEST.VIDS = Cfg.DATALOADER.LOG_DIR+"vids.npy"
Cfg.TEST.CAMIDS = Cfg.DATALOADER.LOG_DIR+"camids.npy"