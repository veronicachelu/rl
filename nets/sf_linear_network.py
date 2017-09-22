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


class SFLinearNetwork:
    def __init__(self, nb_actions, nb_states, scope):
        with tf.variable_scope(scope):

            self.inputs = tf.placeholder(shape=[None, nb_states], dtype=tf.float32, name="Input")
            self.nb_actions = nb_actions
            self.sf_temp = layers.fully_connected(self.inputs, num_outputs=nb_actions * nb_states,
                                               activation_fn=None,
                                               variables_collections=tf.get_collection("variables"),
                                               outputs_collections="activations", scope="sf")
            self.sf = tf.reshape(self.sf_temp, [-1, self.nb_actions, nb_states])

            self.reward = layers.fully_connected(self.inputs, num_outputs=1,
                                                 activation_fn=None,
                                                 variables_collections=tf.get_collection("variables"),
                                                 outputs_collections="activations", scope="reward")
            self.sf_temp = tf.reshape(self.sf, [-1, nb_states])

            self.w_r = tf.tile(tf.expand_dims(tf.expand_dims(tf.squeeze(self.get_var('orig/reward/weights')), 0), 0),
                               [FLAGS.batch_size, nb_actions, 1])
            self.q = tf.reduce_sum(tf.multiply(self.sf, self.w_r), 2)


            if scope != 'target':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
                self.actions_onehot = tf.one_hot(self.actions, nb_actions, dtype=tf.float32, name="actions_one_hot")
                self.actions_onehot = tf.tile(tf.expand_dims(self.actions_onehot, 2), [1, 1, nb_states])
                self.target_sf = tf.placeholder(shape=[None, nb_states], dtype=tf.float32, name="target_SF")
                self.target_reward = tf.placeholder(shape=[None], dtype=tf.float32, name="target_reward")

                self.sf_a = tf.reduce_sum(tf.multiply(self.sf, self.actions_onehot),
                                              reduction_indices=1, name="SF_a")
                # Loss functions
                td_error_sf = self.sf_a - self.target_sf
                mse_reward = self.target_reward - tf.squeeze(self.reward)
                self.sf_loss = tf.reduce_mean(huber_loss(td_error_sf))
                self.reward_loss = tf.reduce_mean(huber_loss(mse_reward))
                self.total_loss = self.sf_loss + self.reward_loss

                if FLAGS.optimizer == "Adam": # to add more optimizers
                    optimizer_sf = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_sf)
                    optimizer_r = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_r)
                else: # default = Adam
                    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
                gradients_sf, self.train_op_sf = minimize_and_clip(optimizer_sf, self.sf_loss, tf.trainable_variables(), FLAGS.gradient_norm_clipping)
                gradients_r, self.train_op_r = minimize_and_clip(optimizer_sf, self.total_loss, tf.trainable_variables(), FLAGS.gradient_norm_clipping)
                # gradients, self.train_op = minimize(optimizer, self.action_value_loss, tf.trainable_variables())
                self.summaries = []
                # self.summaries.append(
                #     tf.contrib.layers.summarize_collection("variables"))  # tf.get_collection("variables")))
                # self.summaries.append(tf.contrib.layers.summarize_collection("activations",
                #                                                              summarizer=tf.contrib.layers.summarize_activation))
                for grad, weight in gradients_sf:
                    if grad is not None:
                        self.summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                        self.summaries.append(tf.summary.histogram(weight.name, weight))

                for grad, weight in gradients_r:
                    if grad is not None:
                        self.summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                        self.summaries.append(tf.summary.histogram(weight.name, weight))

                self.merged_summary = tf.summary.merge(self.summaries)

    def get_var(self, name):
        all_vars = tf.trainable_variables()
        for i in range(len(all_vars)):
            if all_vars[i].name.startswith(name):
                return all_vars[i]
        return None


