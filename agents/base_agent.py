import tensorflow as tf
import numpy as np
from configs import dqn_flags
from collections import deque

FLAGS = tf.app.flags.FLAGS

class BaseAgent:
    def __init__(self, game, sess, nb_actions, global_step):
        self.global_episode = global_step
        self.increment_global_episode = self.global_episode.assign_add(1)
        self.sess = sess
        self.graph = sess.graph
        self.episode_buffer = deque()
        self.actions = np.zeros([nb_actions])
        self.env = game

    def update_target_graph_tao(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign((1 - FLAGS.TAO) * to_var.value() + FLAGS.TAO * from_var.value()))
        return op_holder


    def update_target_graph(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder