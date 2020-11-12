import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import myenv


frame_skip = 10


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


class PddmPolicy:
    def __init__(self, nn, n_trials, horizon):
        self.nn = nn

        self.horizon = horizon
        self.n_trials = n_trials
        self.act_dim = 1

        self.gamma = 1.0
        self.beta = 0.9
        self.sigma = 0.9

        self.reset()

    def __call__(self, s):
        a = [np.zeros((self.n_trials, self.act_dim)) for _ in range(self.horizon)]
        s = np.tile(s, (self.n_trials, 1))

        # Shift action mean
        self.mu[:-1] = self.mu[1:]

        # Sample action candidates
        for i in range(self.horizon):
            eps = np.random.normal(0, self.sigma, size=a[i].shape)

            if i == 0:
                n = self.beta * eps
            else:
                n = self.beta * eps + (1 - self.beta) * n
            a[i] = self.mu[i].reshape((1, -1)) + n

        # Predict rewards for the set of action candidates
        r_total = np.zeros(self.n_trials)
        for i in range(self.horizon):
            x = tf.concat((s, a[i]), axis=1)
            s = self.nn(x).numpy()

            r_action = -0.1 * np.sum(a[i]**2, axis=1)
            r_stable = 10 - 50 * np.abs(s[:, 1])
            r_total += r_stable + r_action

        # Update self.mu using rewards
        weight = np.exp(self.gamma * r_total).reshape((-1, 1)) + 1e-10
        for i in range(self.horizon):
            self.mu[i] = np.sum(a[i] * weight, axis=0) / np.sum(weight)

        return self.mu[0]

    def __call__array(self, s):
        actions = np.zeros((self.n_trials, self.horizon, self.act_dim))
        states = np.zeros((self.n_trials, self.horizon+1, s.shape[0]))
        states[:, 0, :] = np.tile(s, (self.n_trials, 1))

        self.mu[:-1] = self.mu[1:]

        # Sample action candidates
        eps = np.random.normal(0, self.sigma, size=actions.shape)
        for i in range(self.horizon):
            if i == 0:
                n = self.beta * eps[:, i, :]
            else:
                n = self.beta * eps[:, i, :] + (1 - self.beta) * n
            actions[:, i, :] = self.mu[i, :].reshape((1, -1)) + n

        # Predict rewards for each action candidates
        r_total = np.zeros(self.n_trials)
        for i in range(self.horizon):
            x = tf.concat((states[:, i, :], actions[:, i, :]), axis=1)
            states[:, i+1, :] = self.nn(x).numpy()

        r_action = -0.1 * np.sum(actions**2, axis=(1, 2))
        r_stable = 10 - 50 * np.sum(np.abs(states[:, :, 1]), axis=1)
        r_total += r_stable + r_action

        # Update self.mu using rewards
        weight = np.exp(self.gamma * r_total).reshape((-1, 1, 1)) + 1e-10
        self.mu = np.sum(actions * weight, axis=0) / np.sum(weight)

        return self.mu[0, :]

    def reset(self):
        self.mu = np.zeros((self.horizon, self.act_dim))


def train():
    n_epochs       =  40
    batchsize      = 512
    n_train        =  50  # Number of trajectories in each epoch
    rollout_length = 500  # Trajectory length in each trial
    lr = 0.001

    # training args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', default='result', help=('Directory to output trained policies, logs, and plots. A subdirectory is created for each job. This is speficified relative to working directory'))
    parser.add_argument('--use_gpu', action="store_true")
    args = parser.parse_args()

    # Create an output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create environment
    env = myenv.InvertedPendulumEnv(frame_skip=5)

    # Create an NN model
    model = make_model(env.observation_space.shape[0])
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(lr=lr)

    # Setup a scaling factor
    dataset = collect_data(env, n_train, rollout_length)
    observations = dataset[0]
    scale = 1.0 / np.std(observations, axis=0)
    scale = scale.reshape((1, -1))
    print("scale factor is {}.".format(scale))

    train_loss = []
    for epoch in range(n_epochs):
        # Collect training dataset
        train_dataset = collect_data(env, n_train, rollout_length)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(n_train * rollout_length).batch(batchsize)

        # Training
        loss_total = 0.0
        for s, a, s_next in train_dataset:
            with tf.GradientTape() as tape:
                x = tf.concat((s, a), axis=1)
                s_predict = model(x)
                loss = loss_object(s_next*scale, s_predict*scale)

            # Update the model
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_total += float(loss)

        # Print training loss
        train_loss.append(loss_total / (n_train * rollout_length))
        print("epoch {:4d}: train loss = {:.5e}".format(epoch, train_loss[-1]))

    model.save("result/param.h5")
    np.savetxt("result/loss.csv", np.array(train_loss))


def make_model(out_dim):
    model = tf.keras.Sequential([
        layers.Dense(250, activation="relu"),
        layers.Dense(250, activation="relu"),
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
