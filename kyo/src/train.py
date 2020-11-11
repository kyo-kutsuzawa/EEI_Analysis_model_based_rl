# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import myenv


frame_skip = 5


class MpcPolicy:
    def __init__(self, nn, n_trials, horizon):
        self.nn = nn

        self.horizon = horizon
        self.n_trials = n_trials
        self.act_dim = 1

    def __call__(self, s):
        s = np.tile(s, (self.n_trials, 1))
        r_total = np.zeros(self.n_trials)

        for i in range(self.horizon):
            a = np.random.uniform(-4.2, 4.2, size=(self.n_trials, self.act_dim)).astype(np.float64)
            x = tf.concat((s, a), axis=1)
            s = self.nn(x).numpy()

            r_action = -0.1 * np.sum(a**2, axis=1)
            r_stable = 10 - 50 * np.abs(s[:, 1])
            r_total += r_stable + r_action

            if i == 0:
                a0 = a

        idx_best = np.argmax(r_total)
        a_best = a0[idx_best]
        return a_best


def train():
    n_epochs       = 100
    batchsize      = 512
    n_train        =  50
    n_val          =  50
    rollout_length = 100

    # training args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', default='result', help=('Directory to output trained policies, logs, and plots. A subdirectory is created for each job. This is speficified relative to working directory'))
    parser.add_argument('--use_gpu', action="store_true")
    args = parser.parse_args()

    # Create an output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create environment
    env = myenv.InvertedPendulumEnv(frame_skip=5)

    model = make_model(env.observation_space.shape[0])
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    dataset = collect_data(env, n_train, rollout_length)
    observations = dataset[0]
    std = np.std(observations[0:10], axis=0)  # only the first 10 steps are used for calculating std
    std = std.reshape((1, -1))
    print(std)

    for epoch in range(n_epochs):
        # Collect training/validation dataset
        train_dataset = collect_data(env, n_train, rollout_length)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(n_train * rollout_length).batch(batchsize)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        loss_total = 0.0

        for s, a, s_next in train_dataset:
            with tf.GradientTape() as tape:
                x = tf.concat((s, a), axis=1)
                s_predict = model(x)
                loss = loss_object(s_next / std, s_predict / std)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            loss_total += float(loss)

        loss_mean = loss_total / (n_train * rollout_length)
        print("epoch {:4d}: loss = {:.5e}".format(epoch, loss_mean))

    model.save('result/param.h5')


def evaluate():
    n_trials = 2000
    horizon = 20

    env = myenv.InvertedPendulumEnv(frame_skip=1)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = make_model(obs_dim)
    x = np.zeros((n_trials, obs_dim+act_dim))
    model(x)
    model.load_weights("result/param.h5")

    policy = MpcPolicy(model, n_trials, horizon)
    obs = env.reset()
    while True:
        action = policy(obs)

        for i in range(frame_skip):
            env.render()
            obs, _, _, _ = env.step(action)


def evaluate_prediction():
    import matplotlib.pyplot as plt

    env = myenv.InvertedPendulumEnv(frame_skip=frame_skip)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = make_model(obs_dim)
    x = np.zeros((1, obs_dim+act_dim))
    model(x)
    model.load_weights("result/param.h5")

    N = 6
    observations_true = []
    observations_est  = []
    for n in range(N):
        observations_true.append([])
        observations_est.append([])

        obs = env.reset()
        obs_est = obs
        for _ in range(10):
            action = env.action_space.sample()
            obs_true, _, _, _ = env.step(action)

            x = tf.concat((obs_est, action), axis=0)
            x = tf.reshape(x, (1, -1))
            obs_est = model(x).numpy().flatten()

            observations_true[-1].append(obs_true)
            observations_est[-1].append(obs_est.copy())

    observations_true = np.array(observations_true)
    observations_est  = np.array(observations_est)

    fig = plt.figure()
    n_row = np.ceil(np.sqrt(obs_dim))
    n_col = np.ceil(obs_dim/n_row)
    axes = [fig.add_subplot(n_row, n_col, i+1) for i in range(obs_dim)]

    for i, ax in enumerate(axes):
        for n in range(N):
            ax.plot(np.arange(observations_true.shape[1]), observations_true[n, :, i], color="C{}".format(n), ls="--")
            ax.plot(np.arange(observations_est.shape[1]),  observations_est[n, :, i],  color="C{}".format(n))

    fig.tight_layout()
    plt.show()


def make_model(out_dim):
    model = tf.keras.Sequential([
        layers.Dense(250, activation='relu'),
        layers.Dense(250, activation='relu'),
        layers.Dense(out_dim)
    ])

    return model


def collect_data(env, n_rollout, rollout_length):
    observations = []
    actions = []
    observations_next = []

    for _ in range(n_rollout):
        obs = env.reset()

        for _ in range(rollout_length):
            action = env.action_space.sample()

            obs_next, _, _, _ = env.step(action)

            observations.append(obs)
            actions.append(action)
            observations_next.append(obs_next)

            obs = obs_next

    observations      = np.array(observations, dtype=np.float64)
    actions           = np.array(actions, dtype=np.float64)
    observations_next = np.array(observations_next, dtype=np.float64)

    return observations, actions, observations_next


if __name__ == '__main__':
    tf.contrib.eager.enable_eager_execution()
    train()
    evaluate_prediction()
    evaluate()
