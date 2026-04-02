import os

class Config:
    # 数据集配置
    DATA_ROOT = "/home/mdisk2/hulei/MultiView/ADer/data/Real-IAD/realiad_1024/audiojack"
    VIEW_NAMES = ["C1", "C2", "C3", "C4", "C5"]
    MASK_SUFFIX = ".png"

    # 训练参数
    BATCH_SIZE = 1
    NUM_EPOCHS = 100
    LR = 3e-4
    WEIGHT_DECAY = 1e-5
    LR_DECAY_STEP = 20
    LR_DECAY_GAMMA = 0.7

    # 模型参数
    ENCODER = "resnet34"
    PRETRAINED = True
    FEAT_CHANNELS = 256
    ATTENTION_RATIO = 8

    # 硬件设置
    GPUS = [0,1]
    NUM_WORKERS = 8

    # 数据增强
    IMG_SIZE = (512, 512)
    AUG_ROTATION = (-15, 15)
    AUG_SCALE = (0.9, 1.1)
    AUG_BRIGHTNESS = 0.2
    AUG_CONTRAST = 0.2

    # 日志和保存
    LOG_DIR = "./logs"
    SAVE_DIR = "./checkpoints"
    LOG_INTERVAL = 50
    SAVE_INTERVAL = 5

    @classmethod
    def to_dict(cls):
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
