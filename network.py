import numpy as np
import sys
import os

import tensorflow as tf

import features
from config import Config
from util import Utils 

class Network(object):
    def __init__(self, model_name, device):
        self.model_name = model_name 
        self.device = device

        self.img_height = Config.network.INPUT_IMAGE_HEIGHT
        self.img_width = Config.network.INPUT_IMAGE_WIDTH
        # multiply 2, becuase current and next gamestate is used to predict strategy
        self.img_channels = 2 * sum(f.planes for f in features.DEFAULT_FEATURES)

        self.num_strategy = len(Config.strategy.NAMES) 

        self.model_dir = self.get_model_dir(["model_name"])
        self.log_dir = self.get_log_dir(["model_name"])

        self.learning_rate = Config.train.LEARNING_RATE

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self.create_placeholder()
                self.create_network()
                self.create_train_op()

                vars = tf.global_variables()
                self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
                self.sess = tf.Session(
                        graph=self.graph,
                        config=tf.ConfigProto(
                            allow_soft_placement=True,
                            # log_device_placement=False,
                            gpu_options=tf.GPUOptions(allow_growth=True)
                        ))
                self.sess.run(tf.global_variables_initializer()) 
                self.log_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)


    def create_placeholder(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.strategy_index = tf.placeholder(tf.float32, [None, self.num_strategy])
        
        self.global_step = tf.Variable(0, trainable=False, name='step')
        self.is_train = tf.placeholder(tf.bool, name='is_train', shape=[])

        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

    def create_network(self):
        with tf.variable_scope('Network'):
            conv1 = tf.layers.conv2d(self.x, Config.network.NUM_RESIDUAL_FILTERS, [Config.network.FILTER_SIZE, Config.network.FILTER_SIZE], strides=(1, 1), padding='SAME')
            conv1 = tf.nn.relu(tf.contrib.layers.batch_norm(
                                            conv1,
                                            decay=0.9, 
                                            updates_collections=None,
                                            epsilon=1e-5,
                                            scale=True,
                                            center=True,
                                            is_training=self.is_train,
                                            scope='conv1'))
            
        
            # residual block
            current_conv = conv1
            for i in range(Config.network.NUM_RESIDUAL_BLOCKS):
                int_conv = tf.layers.conv2d(current_conv, Config.network.NUM_RESIDUAL_FILTERS, [Config.network.FILTER_SIZE, Config.network.FILTER_SIZE], strides=(1, 1), padding='SAME')
                int_conv = tf.nn.relu(tf.contrib.layers.batch_norm(
                                            int_conv,
                                            decay=0.9, 
                                            updates_collections=None,
                                            epsilon=1e-5,
                                            scale=True,
                                            center=True,
                                            is_training=self.is_train,
                                            scope='int_conv'+ str(i)))
                
                out_conv = tf.layers.conv2d(int_conv, Config.network.NUM_RESIDUAL_FILTERS, [Config.network.FILTER_SIZE, Config.network.FILTER_SIZE], strides=(1, 1), padding='SAME')
                out_conv = tf.contrib.layers.batch_norm(
                                            out_conv,
                                            decay=0.9, 
                                            updates_collections=None,
                                            epsilon=1e-5,
                                            scale=True,
                                            center=True,
                                            is_training=self.is_train,
                                            scope='out_conv'+ str(i))
                # skip connection
                out_conv = tf.nn.relu(current_conv + out_conv)
                current_conv = out_conv

            # head
            conv_final = tf.layers.conv2d(current_conv, 1, [1, 1], strides=(1, 1), padding='SAME')
            conv_final = tf.nn.relu(tf.contrib.layers.batch_norm(
                    conv_final,
                    decay=0.9, 
                    updates_collections=None,
                    epsilon=1e-5,
                    scale=True,
                    center=True,
                    is_training=self.is_train,
                    scope='conv_final'))
            
            flat = tf.reshape(conv_final, [-1, self.img_height * self.img_width])
            fc1 = tf.layers.dense(flat, 256, activation=tf.nn.relu, name='fc1')
            self.logits = tf.layers.dense(fc1, self.num_strategy, activation=None, name='logits_v')
            self.softmax = tf.nn.softmax(self.logits)

    def create_train_op(self):
        self.opt = tf.train.MomentumOptimizer(
            learning_rate=self.var_learning_rate, momentum=0.9, use_nesterov=True)

        # loss
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.strategy_index))
        # Regularizer
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        reg_variables = tf.trainable_variables()
        self.reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

        self.cost_reg = self.cost + self.reg_term

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.train_op = self.opt.minimize(self.cost_reg, global_step=self.global_step)        

    def get_model_dir(self, attr):
        model_dir = os.path.join('./', Config.resource.MODEL_CHECKPOINT_DIR)
        for attr in attr:
            if hasattr(self, attr):
                if attr == "model_name":
                    model_dir += "/%s" % (getattr(self, attr))
                else:
                    model_dir += "/%s=%s" % (attr, getattr(self, attr))
        return model_dir

    def get_log_dir(self, attr):
        model_dir = os.path.join('/', Config.resource.MODEL_LOG_DIR)
        for attr in attr:
            if hasattr(self, attr):
                if attr == "model_name":
                    model_dir += "/%s" % (getattr(self, attr))
                else:
                    model_dir += "/%s=%s" % (attr, getattr(self, attr))
        return model_dir

    def get_base_feed_dict(self):
        return {self.var_learning_rate: self.learning_rate}

    def save(self, directory, step, model_name='checkpoint'):
        print(" [*] Saving checkpoints...")
        save_model_dir = directory
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        self.saver.save(self.sess, os.path.join(save_model_dir, model_name), global_step = step)

    def load(self, load_model_dir):
        print(" [*] Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(load_model_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(load_model_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
        else:
            print(" [!] Load FAILED: %s" % load_model_dir)

    def predict_single(self, x, is_train):
        feed_dict = {self.x: x[None, :], self.is_train: is_train}
        prediction = self.sess.run(self.softmax,
             feed_dict=feed_dict)
        return prediction[0]

    def predict(self, x, is_train):
        feed_dict = {self.x: x, self.is_train: is_train}
        prediction = self.sess.run(self.softmax,
             feed_dict=feed_dict)
        return prediction

    def get_cost(self, x, strategy_index, is_train):
        feed_dict = {self.x: x, self.strategy_index: strategy_index,
                    self.is_train: is_train}
        cost = self.sess.run(self.cost,
             feed_dict=feed_dict)
        return cost

    def train(self, x, strategy_index):
        feed_dict = self.get_base_feed_dict()
        feed_dict.update({
            self.x: x,
            self.strategy_index: strategy_index,
            self.is_train: True
        })
        _, cost, reg_term = \
            self.sess.run([self.train_op, self.cost, self.reg_term],
                     feed_dict=feed_dict)
        return cost, reg_term