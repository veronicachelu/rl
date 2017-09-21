import threading
import multiprocessing
from threading import Lock
import gym
from gym import wrappers
import tensorflow as tf
from agents.dqn_linear_agent import DQNLinearAgent
from env_wrappers.txt_wrapper import GridWorld
from nets.dqn_linear_network import DQLinearNetwork
from tensorflow.python import debug as tf_debug
from configs import dqn_linear_flags
import os
FLAGS = tf.app.flags.FLAGS

main_lock = Lock()

class DQNLinear:
    def __init__(self):
        sess = tf.Session()
        with sess:
            global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

            env = GridWorld(load_path=FLAGS.mdp_path)
            nb_actions = env.nb_actions

            self.agent = DQNLinearAgent(env, sess, nb_actions, global_step)
            self.saver = tf.train.Saver(max_to_keep=1000)

        if FLAGS.resume or not FLAGS.train:
            checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.algorithm)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            print("Loading Model from {}".format(ckpt.model_checkpoint_path))
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

    def eval(self):
        self.agent.eval(self.saver)

    def train(self):
        self.agent.play(self.saver)


