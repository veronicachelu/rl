import threading
import multiprocessing
from threading import Lock
import gym
from gym import wrappers
import gym_fast_envs
# import gym_ple
import tensorflow as tf
from agents.categorical_dqn_agent import CategoricalDQNAgent
from env_wrappers.atari_environment import AtariEnvironment
from nets.categorical_dqn_network import CategoricalDQNetwork
from tensorflow.python import debug as tf_debug
from configs import categorical_dqn_flags
import os
FLAGS = tf.app.flags.FLAGS

main_lock = Lock()

class CategoricalDQN:
    def __init__(self):
        sess = tf.Session()
        with sess:
            global_step = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)

            self.env = gym_env = gym.make(FLAGS.game)
            if FLAGS.seed and FLAGS.seed != -1:
                gym_env.seed(FLAGS.seed)

            if FLAGS.monitor:
                gym_env = gym.wrappers.Monitor(gym_env, FLAGS.experiments_dir)

            env = AtariEnvironment(gym_env=gym_env, resized_width=FLAGS.resized_width,
                                   resized_height=FLAGS.resized_height,
                                   agent_history_length=FLAGS.agent_history_length)
            nb_actions = len(env.gym_actions)

            self.agent = CategoricalDQNAgent(env, sess, nb_actions, global_step)
            self.saver = tf.train.Saver(max_to_keep=1000)

        if FLAGS.resume:
            checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.algorithm)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            print("Loading Model from {}".format(ckpt.model_checkpoint_path))
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

    def train(self):
        self.agent.play(self.saver)

        while True:
            if FLAGS.show_training:
                self.env.render()
