import numpy as np

from config import Config 

STEP_H_TO_X = {
    "PlayGround" :Config.play_ground.PLAYAREA_HEIGHT / (Config.network.INPUT_IMAGE_HEIGHT),
    }

STEP_W_TO_Y = {
    "PlayGround": Config.play_ground.PLAYAREA_WIDTH / (Config.network.INPUT_IMAGE_WIDTH),
    }

class PlayGroundUtils(object):    
    @staticmethod
    def hw_to_xy(h, w):
        assert(0 <= h and h <= Config.network.INPUT_IMAGE_HEIGHT - 1)
        assert(0 <= w and w <= Config.network.INPUT_IMAGE_WIDTH - 1)
        x = Config.play_ground.X_PLAYAREA_MIN + h * STEP_H_TO_X["PlayGround"] + 0.5 * STEP_H_TO_X["PlayGround"]
        y = Config.play_ground.Y_PLAYAREA_MIN + w * STEP_W_TO_Y["PlayGround"] + 0.5 * STEP_W_TO_Y["PlayGround"]
        return x, y

    @staticmethod
    def xy_to_hw(x, y):
        assert(Config.play_ground.X_PLAYAREA_MIN <= x and x <= Config.play_ground.X_PLAYAREA_MAX)
        assert(Config.play_ground.Y_PLAYAREA_MIN - Config.play_ground.STONE_RADIUS <= y and y <= Config.play_ground.Y_PLAYAREA_MAX)
        if x == Config.play_ground.X_PLAYAREA_MAX:
            h = Config.network.INPUT_IMAGE_HEIGHT - 1
        else:
            h = int((x - Config.play_ground.X_PLAYAREA_MIN) / STEP_H_TO_X["PlayGround"]) 
        if y == Config.play_ground.Y_PLAYAREA_MAX:
            w = Config.network.INPUT_IMAGE_WIDTH - 1
        else:
            w = int((y - Config.play_ground.Y_PLAYAREA_MIN) / STEP_W_TO_Y["PlayGround"])
        return h, w

    @staticmethod
    def is_in_playarea(x, y):
        return (Config.play_ground.X_PLAYAREA_MIN + Config.play_ground.STONE_RADIUS < x) and (Config.play_ground.X_PLAYAREA_MAX - Config.play_ground.STONE_RADIUS > x) \
            and (Config.play_ground.Y_PLAYAREA_MIN - Config.play_ground.STONE_RADIUS < y) and (Config.play_ground.Y_PLAYAREA_MAX - Config.play_ground.STONE_RADIUS > y)

    @staticmethod
    def is_in_house(x, y):
        rx = x - Config.play_ground.TEE_X
        ry = y - Config.play_ground.TEE_Y
        R = np.sqrt(rx**2 + ry**2) 
        return R < Config.play_ground.HOUSE_RADIUS + Config.play_ground.STONE_RADIUS

    @staticmethod
    def is_in_red_circle(x, y):
        rx = x - Config.play_ground.TEE_X
        ry = y - Config.play_ground.TEE_Y
        R = np.sqrt(rx**2 + ry**2) 
        return R < 0.333 * Config.play_ground.HOUSE_RADIUS + Config.play_ground.STONE_RADIUS

    @staticmethod
    def is_in_guardzone(x, y):
        if not PlayGroundUtils.is_in_playarea(x, y):
            return False
        if PlayGroundUtils.is_in_house(x, y):
            return False
        if y < Config.play_ground.TEE_Y + Config.play_ground.STONE_RADIUS:
            return False
        if x < Config.play_ground.TEE_X - 0.5 * Config.play_ground.HOUSE_RADIUS - Config.play_ground.STONE_RADIUS \
            or x > Config.play_ground.TEE_X + 0.5 * Config.play_ground.HOUSE_RADIUS + Config.play_ground.STONE_RADIUS:
            return False
        return True

class StrategyUtils(object):    
    @staticmethod
    def name_to_idx(name):
        return Config.strategy.NAMES.index(name) 

    @staticmethod
    def idx_to_name(idx):
        return Config.strategy.NAMES[idx]

class Utils(object):
    # value = ValueUtils
    # policy = PolicyUtils
    play_ground = PlayGroundUtils
    strategy = StrategyUtils
