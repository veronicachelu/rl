import math
from math import sqrt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import tensorflow.contrib.layers as layers
from utils.tf_util import layer_norm_fn
from utils.optimizers import huber_loss, minimize_and_clip, l2_loss, minimize
FLAGS = tf.app.flags.FLAGS


class DQLinearNetwork:
    def __init__(self, nb_actions, nb_states, scope):
        with tf.variable_scope(scope):

            self.inputs = tf.placeholder(shape=[None, nb_states], dtype=tf.float32, name="Input")
            self.nb_actions = nb_actions
            self.action_values = value_out = layers.fully_connected(self.inputs, num_outputs=nb_actions,
                                               activation_fn=None,
                                               variables_collections=tf.get_collection("variables"),
                                               outputs_collections="activations")

            if scope != 'target':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
                self.actions_onehot = tf.one_hot(self.actions, nb_actions, dtype=tf.float32, name="actions_one_hot")
                self.target_q = tf.placeholder(shape=[None], dtype=tf.float32, name="target_Q")

                self.action_value = tf.reduce_sum(tf.multiply(self.action_values, self.actions_onehot),
                                                  reduction_indices=1, name="Q")
                # Loss functions
                td_error = self.action_value - self.target_q
                self.action_value_loss = tf.reduce_mean(huber_loss(td_error))
                #self.action_value_loss = l2_loss(td_error)
                if FLAGS.optimizer == "Adam": # to add more optimizers
                    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
                else: # default = Adam
                    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
                gradients, self.train_op = minimize_and_clip(optimizer, self.action_value_loss, tf.trainable_variables(), FLAGS.gradient_norm_clipping)
                # gradients, self.train_op = minimize(optimizer, self.action_value_loss, tf.trainable_variables())
                self.summaries = []
                # self.summaries.append(
                #     tf.contrib.layers.summarize_collection("variables"))  # tf.get_collection("variables")))
                # self.summaries.append(tf.contrib.layers.summarize_collection("activations",
                #                                                              summarizer=tf.contrib.layers.summarize_activation))

                for grad, weight in gradients:
                    if grad is not None:
                        self.summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                        self.summaries.append(tf.summary.histogram(weight.name, weight))

                self.merged_summary = tf.summary.merge(self.summaries)




