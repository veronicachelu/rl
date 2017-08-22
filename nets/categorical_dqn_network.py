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


class CategoricalDQNetwork:
    def __init__(self, nb_actions, scope):
        with tf.variable_scope(scope):

            self.inputs = tf.placeholder(
                shape=[None, FLAGS.resized_height, FLAGS.resized_width, FLAGS.agent_history_length], dtype=tf.float32,
                name="Input")

            self.image_summaries = []
            self.image_summaries.append(
                tf.summary.image('input', tf.expand_dims(self.inputs[:, :, :, 0], 3), max_outputs=FLAGS.batch_size))

            out = self.inputs
            self.nb_actions = nb_actions
            self.out_size = self.nb_actions * FLAGS.nb_atoms

            with tf.variable_scope("convnet"):
                out = layers.conv2d(out, num_outputs=32, kernel_size=5, stride=2, activation_fn=tf.nn.relu,
                                      variables_collections=tf.get_collection("variables"),
                                      outputs_collections="activations", scope="conv1")

                out = layers.conv2d(out, num_outputs=32, kernel_size=5, stride=2, activation_fn=tf.nn.relu,
                                      padding="VALID",
                                      variables_collections=tf.get_collection("variables"),
                                      outputs_collections="activations", scope="conv2")
            conv_out = layers.flatten(out)

            with tf.variable_scope("action_value"):
                value_out = layers.fully_connected(conv_out, num_outputs=FLAGS.hidden_size,
                                                   activation_fn=None,
                                                   variables_collections=tf.get_collection("variables"),
                                                   outputs_collections="activations", scope="fc")
                if FLAGS.layer_norm:
                    value_out = layer_norm_fn(value_out, relu=True)
                else:
                    value_out = tf.nn.relu(value_out)
                value_out = layers.fully_connected(value_out, num_outputs=self.out_size,
                                                   activation_fn=None,
                                                   variables_collections=tf.get_collection("variables"),
                                                   outputs_collections="activations", scope="action_values")
                self.action_values = tf.reshape(value_out, [-1, self.nb_actions, FLAGS.nb_atoms])
                # value_out = tf.transpose(value_out, [2, 0, 1])
                # value_out = tf.map_fn(lambda v: tf.nn.softmax(v), value_out)
                value_out = tf.split(self.action_values, num_or_size_splits=self.nb_actions, axis=1)
                self.action_values_soft = tf.concat(list(map(lambda v: tf.nn.softmax(v, name="action_value_soft"), value_out)), 1)

            if scope != 'target':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
                self.actions_onehot = tf.one_hot(self.actions, self.nb_actions, dtype=tf.float32, name="actions_one_hot")
                self.target_q = tf.placeholder(shape=[None, FLAGS.nb_atoms], dtype=tf.float32, name="target_Q")

                self.actions_onehot = tf.tile(tf.expand_dims(self.actions_onehot, 2), [1, 1, FLAGS.nb_atoms])
                # self.actions_onehot = tf.reshape(self.actions_onehot, [-1, self.nb_actions, FLAGS.nb_atoms])
                self.action_value = tf.reduce_sum(tf.multiply(self.action_values, self.actions_onehot),
                                                  reduction_indices=1, name="Q")
                # Loss functions
                self.action_value_loss = -tf.reduce_sum(tf.multiply(self.target_q, tf.nn.log_softmax(self.action_value)))

                if FLAGS.optimizer == "Adam": # to add more optimizers
                    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
                else: # default = Adam
                    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
                gradients, self.train_op = minimize_and_clip(optimizer, self.action_value_loss, tf.trainable_variables(), FLAGS.gradient_norm_clipping)
                # gradients, self.train_op = minimize(optimizer, self.action_value_loss, tf.trainable_variables())
                self.summaries = []
                self.summaries.append(
                    tf.contrib.layers.summarize_collection("variables"))  # tf.get_collection("variables")))
                self.summaries.append(tf.contrib.layers.summarize_collection("activations",
                                                                             summarizer=tf.contrib.layers.summarize_activation))
                summary_action_value = tf.contrib.layers.summarize_activation(self.action_value)
                self.summaries.append(summary_action_value)
                summary_action_values_soft = tf.contrib.layers.summarize_activation(self.action_values_soft)
                self.summaries.append(summary_action_values_soft)

                for grad, weight in gradients:
                    if grad is not None:
                        self.summaries.append(tf.summary.histogram(weight.name + '_grad', grad))
                        self.summaries.append(tf.summary.histogram(weight.name, weight))

                self.merged_summary = tf.summary.merge(self.summaries)




