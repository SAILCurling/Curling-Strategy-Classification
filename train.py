import numpy as np
import argparse
import time

from tqdm import tqdm
import tensorflow as tf

from network import Network
from features import extract_planes
from memory import Memory
from dcl_utils import DclParser
from config import Config
from util import Utils 

# DATA_DIR = "./labeled_Log/"
DATA_DIR = "../labeled_Logs/"


class TrainerSL:
    def __init__(self, args):
        self.device = args.device

        self.model = Network(args.name, self.device)

        self.num_strategy = len(Config.strategy.NAMES) 

        self.avg_loss = []
        self.avg_acc = []
        self.avg_reg_term = []
        self.time_start = None

        dcl_paths = DclParser.walk_all_dcls(DATA_DIR)
        # sort dcl_paths in order
        dcl_paths = sorted(dcl_paths, key = lambda k:
           int(k.split('[')[-1].split(']')[0].replace('-','').replace(' ','')))
        print('last dcl path=', dcl_paths[-1])

        self.memory = Memory(Config.train.MEMORY_SIZE)
        print("Parsing dcl files ...")
        self.update_memory(dcl_paths)

    def update_memory(self, dcl_paths):
        for dcl_path in tqdm(dcl_paths):
            game_record = DclParser.read_dcl(dcl_path, read_valid_record=False)
            try:
                for i in range(len(game_record) - 1):
                    if game_record[i]['ShotNum'] != 15: # ignore last shot
                        if game_record[i]['ShotName'] != 'NOTDEFINED':
                            game_record[i]['ShotName'] = Utils.strategy.name_to_idx(game_record[i]['ShotName'])
                            self.memory.add(game_record[i], game_record[i + 1])                            
            except:
                print('Failed to read dcl.')

    def get_batch_from_memory(self, type):
        cur_bodies, cur_ShotNums, cur_WhiteToMoves, next_bodies, next_ShotNums, next_WhiteToMoves, strategies = \
                self.memory.sample_batch(type)

        planes_batch = []
        strategy_index_batch = []
        for i in range(Config.train.BATCH_SIZE):
            # make end_score_index (one-hot vector)
            strategy_index = np.zeros(len(Config.strategy.NAMES))
            strategy_index[strategies[i]] = 1.
            strategy_index_batch.append(strategy_index)

            symmetry = np.random.choice(["noop", "reflect"])
            if symmetry == "reflect":
                # body
                for t in range(cur_ShotNums[i]):
                    if Utils.play_ground.is_in_playarea(*cur_bodies[i][t]):
                        cur_bodies[i][t][0] = Config.play_ground.X_PLAYAREA_MAX - cur_bodies[i][t][0] + Config.play_ground.X_PLAYAREA_MIN

                for t in range(next_ShotNums[i]):
                    if Utils.play_ground.is_in_playarea(*next_bodies[i][t]):
                        next_bodies[i][t][0] = Config.play_ground.X_PLAYAREA_MAX - next_bodies[i][t][0] + Config.play_ground.X_PLAYAREA_MIN


            # planes
            planes = extract_planes(
                {
                    "body": cur_bodies[i],
                    "ShotNum": cur_ShotNums[i],
                    "WhiteToMove": cur_WhiteToMoves[i]
                },
                {
                    "body": next_bodies[i],
                    "ShotNum": next_ShotNums[i],
                    "WhiteToMove": next_WhiteToMoves[i]
                },
            )
            planes_batch.append(planes)

        return planes_batch, strategy_index_batch

    def train_one_step(self):
        planes_batch, strategy_index_batch = \
                self.get_batch_from_memory(type='train')
        cost, reg_term = \
            self.model.train(planes_batch, strategy_index_batch)
        prediction = self.model.predict(planes_batch, is_train=True)
        prediction = np.argmax(prediction, axis=1)
        acc = float(sum(prediction == np.argmax(strategy_index_batch, axis=1))) / len(planes_batch)
        steps = tf.train.global_step(self.model.sess, self.model.global_step)
        self.avg_acc.append(acc)
        self.avg_loss.append(cost)
        self.avg_reg_term.append(reg_term)
        if steps % 100 == 0:
            time_end = time.time()
            speed = 0
            if self.time_start:
                elapsed = time_end - self.time_start
                speed = Config.train.BATCH_SIZE * (100.0 / elapsed)
            avg_loss = np.mean(self.avg_loss or [0])
            avg_acc = np.mean(self.avg_acc or [0])
            avg_reg_term = np.mean(self.avg_reg_term or [0])
            print("step {}, acc={:g}, loss={:g}, reg={:g} total={:g} ({:g} pos/s)".format(
                steps, avg_acc, avg_loss, avg_reg_term,
                avg_loss + avg_reg_term,
                speed))
            train_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="train/acc", simple_value=avg_acc),
                tf.Summary.Value(tag="train/loss", simple_value=avg_loss),
                tf.Summary.Value(tag="train/reg_term", simple_value=avg_reg_term),
                ])
            self.model.log_writer.add_summary(train_summaries, steps)
            self.time_start = time_end
            self.avg_acc, self.avg_loss, self.avg_reg_term = [], [], []

        if steps % 1000 == 0:
            # test
            test_avg_acc = 0.
            test_avg_loss = 0.

            for _ in range(0 ,10):
                planes_batch, strategy_index_batch = \
                    self.get_batch_from_memory(type='test')
                cost = self.model.get_cost(planes_batch, strategy_index_batch, is_train=False)
                prediction = self.model.predict(planes_batch, is_train=False)
                prediction = np.argmax(prediction, axis=1)
                acc = float(sum(prediction == np.argmax(strategy_index_batch, axis=1))) / len(planes_batch)
                test_avg_acc += acc
                test_avg_loss += cost
            test_avg_acc /= 10.
            test_avg_loss /= 10.
            test_summaries = tf.Summary(value=[
                tf.Summary.Value(tag="test/acc", simple_value=test_avg_acc),
                tf.Summary.Value(tag="test/loss", simple_value=test_avg_loss),
                ])
            self.model.log_writer.add_summary(test_summaries, steps)

        if steps % 5000 == 0:
            self.model.save(self.model.model_dir, steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trainer',
        epilog="Available features are: board, turn_num, end, relative_score_till_last_end, in_house and order_to_tee.")
    parser.add_argument("--name", "-n", help="network name for model", required=True)
    parser.add_argument("--device", "-d", help="device for training the neural network model", default='gpu:0')
    args = parser.parse_args()

    trainer = TrainerSL(args)

    print("# of current data=", trainer.memory.count)
    while True:
        trainer.train_one_step()
      