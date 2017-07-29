from threading import Lock

import numpy as np
import tensorflow as tf
from agents.base_agent import BaseAgent
from nets.dqn_network import DQNetwork
from configs import dqn_flags
from collections import deque
from utils.schedules import LinearSchedule
import os
FLAGS = tf.app.flags.FLAGS
import random

# Starting threads
main_lock = Lock()

class DQNAgent(BaseAgent):
    def __init__(self, game, sess, nb_actions, global_step):
        BaseAgent.__init__(self, game, sess, nb_actions, global_step)
        self.name = "DQN_agent"
        self.model_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.algorithm)

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.exploration = LinearSchedule(FLAGS.explore_steps, FLAGS.final_random_action_prob,
                                          FLAGS.initial_random_action_prob)
        self.summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, FLAGS.algorithm))
        self.summary = tf.Summary()

        self.q_net = DQNetwork(nb_actions, 'orig')
        self.target_net = DQNetwork(nb_actions, 'target')

        self.targetOps = self.update_target_graph('orig', 'target')

        self.probability_of_random_action = self.exploration.value(0)

    def train(self):
        minibatch = random.sample(self.episode_buffer, FLAGS.batch_size)
        rollout = np.array(minibatch)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        done = rollout[:, 4]

        target_actionv_values_evaled = self.sess.run(self.target_net.action_values,
                                                     feed_dict={self.target_net.inputs: np.stack(next_observations, axis=0)})
        target_actionv_values_evaled_max = np.max(target_actionv_values_evaled, axis=1)

        target_actionv_values_evaled_new = []

        for i in range(FLAGS.batch_size):
            if done[i]:
                target_actionv_values_evaled_new.append(rewards[i])
            else:
                target_actionv_values_evaled_new.append(
                    rewards[i] + FLAGS.gamma * target_actionv_values_evaled_max[i])

        feed_dict = {self.q_net.target_q: target_actionv_values_evaled_new,
                     self.q_net.inputs: np.stack(observations, axis=0),
                     self.q_net.actions: actions}
        l, _, ms, img_summ = self.sess.run(
            [self.q_net.action_value_loss,
             self.q_net.train_op,
             self.q_net.merged_summary,
             self.q_net.image_summaries],
            feed_dict=feed_dict)

        self.updateTarget()

        return l / len(rollout), ms, img_summ

    def updateTarget(self):
        for op in self.targetOps:
            self.sess.run(op)

    def play(self, saver):
        episode_count = self.sess.run(self.global_episode)
        total_steps = 0

        print("Starting agent")
        with self.sess.as_default(), self.graph.as_default():
            while total_steps < FLAGS.max_total_steps:
                if total_steps % FLAGS.target_update_freq == 0:
                    self.updateTarget()
                episode_reward = 0
                episode_step_count = 0
                q_values = []
                d = False
                self.probability_of_random_action = self.exploration.value(episode_count)
                s = self.env.get_initial_state()

                while not d:
                    if random.random() <= self.probability_of_random_action:
                        a = np.random.choice(range(len(self.env.gym_actions)))
                    else:
                        feed_dict = {self.q_net.inputs: [s]}
                        action_values_evaled = self.sess.run(self.q_net.action_values, feed_dict=feed_dict)[0]
                        q_values.append(action_values_evaled)
                        a = np.argmax(action_values_evaled)

                    s1, r, d, info = self.env.step(a)

                    r = np.clip(r, -1, 1)
                    episode_reward += r
                    episode_step_count += 1
                    total_steps += 1
                    self.episode_buffer.append([s, a, r, s1, d])

                    if len(self.episode_buffer) == FLAGS.memory_size:
                        self.episode_buffer.popleft()

                    if total_steps > FLAGS.observation_steps and total_steps % FLAGS.update_freq == 0:
                        l, ms, img_summ = self.train()



                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                if len(q_values):
                    self.episode_mean_values.append(np.max(np.asarray(q_values)))

                if episode_count % FLAGS.summary_interval == 0 and episode_count != 0 and episode_count > FLAGS.observation_steps:
                    if episode_count % FLAGS.checkpoint_interval == 0:
                        self.save_model(saver, episode_count)

                    mean_reward = np.mean(self.episode_rewards[-FLAGS.summary_interval:])
                    mean_length = np.mean(self.episode_lengths[-FLAGS.summary_interval:])
                    mean_value = np.mean(self.episode_mean_values[-FLAGS.summary_interval:])

                    # if episode_count % FLAGS.test_performance_interval == 0:
                    #     won_games = self.episode_rewards[-FLAGS.test_performance_interval:].count(1)
                    #     self.summary.value.add(tag='Perf/Won Games/1000', simple_value=float(won_games))

                    self.summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    self.summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    self.summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    self.summary.value.add(tag='Losses/Loss', simple_value=float(l))

                    self.write_summary(ms, img_summ, episode_count)

                self.sess.run(self.increment_global_episode)
                episode_count += 1
