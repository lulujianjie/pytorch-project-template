from yacs.config import CfgNode as cfg
#config tree
Cfg = cfg()

Cfg.DATALOADER = cfg()
Cfg.DATALOADER.LOG_DIR = "./log/" #log dir and saved model dir
Cfg.DATALOADER.DATALOADER_NUM_WORKERS = 8
Cfg.DATALOADER.DATA_DIR = "/data/lujj/datasets/XXX/"

Cfg.MODEL = cfg()
Cfg.MODEL.INPUT_SIZE = [256, 256] #HxW
Cfg.MODEL.DEVICE_ID = "5,6,8"#

Cfg.LOSS = cfg()
Cfg.LOSS.TYPE = 'softmax'

Cfg.SOLVER = cfg()
Cfg.SOLVER.BATCHSIZE = 64
Cfg.SOLVER.OPTIMIZER = 'Adam'
Cfg.SOLVER.BASE_LR = 0.001

Cfg.TEST = cfg()
Cfg.TEST.IMS_PER_BATCH = 128
Cfg.TEST.WEIGHT = ''