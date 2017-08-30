import tensorflow as tf
import numpy as np
from configs import base_flags
from collections import deque

FLAGS = tf.app.flags.FLAGS

class BaseAgent:
    def __init__(self, game, sess, nb_actions, global_step):
        self.global_step = global_step
        self.increment_global_step = self.global_step.assign_add(1)
        self.sess = sess
        self.graph = sess.graph
        self.episode_buffer = deque()
        self.nb_actions = nb_actions
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

    def save_model(self, saver, episode_count):
        saver.save(self.sess, self.model_path + '/model-' + str(episode_count) + '.cptk',
                   global_step=self.global_step)

        print("Saved Model at {}".format(self.model_path + '/model-' + str(episode_count) + '.cptk'))

    def write_summary(self, ms, img_summ):
        summaries = tf.Summary().FromString(ms)
        sub_summaries_dict = {}
        for value in summaries.value:
            value_field = value.WhichOneof('value')
            value_ifo = sub_summaries_dict.setdefault(value.tag,
                                                      {'value_field': None, 'values': []})
            if not value_ifo['value_field']:
                value_ifo['value_field'] = value_field
            else:
                assert value_ifo['value_field'] == value_field
            value_ifo['values'].append(getattr(value, value_field))

        for name, value_ifo in sub_summaries_dict.items():
            summary_value = self.summary.value.add()
            summary_value.tag = name
            if value_ifo['value_field'] == 'histo':
                values = value_ifo['values']
                summary_value.histo.min = min([x.min for x in values])
                summary_value.histo.max = max([x.max for x in values])
                summary_value.histo.num = sum([x.num for x in values])
                summary_value.histo.sum = sum([x.sum for x in values])
                summary_value.histo.sum_squares = sum([x.sum_squares for x in values])
                for lim in values[0].bucket_limit:
                    summary_value.histo.bucket_limit.append(lim)
                for bucket in values[0].bucket:
                    summary_value.histo.bucket.append(bucket)
            else:
                print(
                    'Warning: could not aggregate summary of type {}'.format(value_ifo['value_field']))
        for s in img_summ:
            self.summary_writer.add_summary(s, self.total_steps)
        self.summary_writer.add_summary(self.summary, self.total_steps)
        self.summary_writer.flush()