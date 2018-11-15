import os

def _project_dir():
    d = os.path.dirname
    return d(os.path.abspath(__file__))

# def _data_dir():
    # return os.path.join(_project_dir(), "data")


class ResourceConfig:
    PROJECT_DIR = _project_dir()
    # model
    MODEL_DIR = os.path.join(PROJECT_DIR, "model")
    MODEL_CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
    MODEL_LOG_DIR = os.path.join(MODEL_DIR, "logs")

def create_directories():
    print(' [*] create directories ...')
    dirs = [
        ResourceConfig.PROJECT_DIR,
        ResourceConfig.MODEL_DIR,
        ResourceConfig.MODEL_CHECKPOINT_DIR,
        ResourceConfig.MODEL_LOG_DIR,
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
create_directories() 

class TrainingConfig(object):
    LEARNING_RATE = 0.001
    BATCH_SIZE = 256
    MEMORY_SIZE = 5000000

    DEVICE = "gpu:0"

class NetworkConfig:
    INPUT_IMAGE_HEIGHT = 32
    INPUT_IMAGE_WIDTH = 32 

    OUTPUT_IMAGE_HEIGHT = 32 
    OUTPUT_IMAGE_WIDTH = 32

    NUM_RESIDUAL_FILTERS = 32
    NUM_RESIDUAL_BLOCKS = 9
    FILTER_SIZE = 3

class PlayGroundConfig:
    """GAME Configuration"""
    # Execution uncertainty is modeled by asymmetric gaussian noise
    # std of x: RAND * 0.5 
    # std of y: RAND * 2.0 
    RAND = 0.145 
    STONE_RADIUS = 0.145
    HOUSE_RADIUS = 1.83

    X_PLAYAREA_MIN = 0
    X_PLAYAREA_MAX = 4.75
    Y_PLAYAREA_MIN = 3.05
    Y_PLAYAREA_MAX = 3.05 + 8.23
    
    PLAYAREA_HEIGHT = X_PLAYAREA_MAX - X_PLAYAREA_MIN
    PLAYAREA_WIDTH = Y_PLAYAREA_MAX - Y_PLAYAREA_MIN
    
    TEE_X = 2.375
    TEE_Y = 4.88 

class StrategyConfig:
    NAMES = [
        "TripleTakeout",
        "DoubleTakeout",
        "Takeout", 
        "Peel", 
        "Guard", 
        "ComeAround", 
        "Draw", 
        "ThrowAway", 
        "Freeze"
    ]

class Config:
    """
    """
    # VERSION = __VERSION__

    resource = ResourceConfig
    train = TrainingConfig
    play_ground = PlayGroundConfig
    # policy = PolicyConfig
    # value = ValueConfig
    network = NetworkConfig
    strategy = StrategyConfig