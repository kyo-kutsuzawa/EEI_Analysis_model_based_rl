import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import myenv
import tqdm


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

        self.gamma = 10.0
        self.beta = 0.99
        self.sigma = 0.3

        self.reset()

    def __call__(self, s):
        a = [np.zeros((self.n_trials, self.act_dim)) for _ in range(self.horizon)]
        s = np.tile(s, (self.n_trials, 1))

        # Shift action mean
        a_past = self.mu[0].copy()
        self.mu[:-1] = self.mu[1:]

        # Sample action candidates
        for i in range(self.horizon):
            eps = np.random.normal(0, self.sigma, size=a[i].shape)

            #if i == 0:
            #    n = self.beta * eps
            #else:
            #    n = self.beta * eps + (1 - self.beta) * n
            #a[i] = self.mu[i].reshape((1, -1)) + n
            if i == 0:
                a[i] = self.beta * (self.mu[i].reshape((1, -1)) + eps) + (1 - self.beta) * a_past
            else:
                a[i] = self.beta * (self.mu[i].reshape((1, -1)) + eps) + (1 - self.beta) * a[i-1]

        # Predict rewards for the set of action candidates
        r_total = np.zeros(self.n_trials)
        for i in range(self.horizon):
            x = tf.concat((s, a[i]), axis=1)
            s = self.nn(x).numpy()

            r_action = -0.1 * np.sum(a[i]**2, axis=1)
            r_stable = 10 - 50 * np.abs(s[:, 1])
            r_total += r_stable + r_action

        # Update self.mu using rewards
        #weight = np.exp(self.gamma * r_total).reshape((-1, 1))
        weight = np.exp(self.gamma * (r_total - np.max(r_total))).reshape((-1, 1))
        for i in range(self.horizon):
            self.mu[i] = np.sum(a[i] * weight, axis=0) / (np.sum(weight) + 1e-10)

        return self.mu[0]

    def reset(self):
        self.mu = np.zeros((self.horizon, self.act_dim))


def train():
    # Setup constant values
    n_epochs       =  40
    batchsize      = 512
    n_train        = int( 50 * frame_skip)  # Number of trajectories in each epoch
    rollout_length = int(200 / frame_skip)  # Trajectory length in each trial
    lr = 0.001  # Learning rate

    # training args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', default='result', help=('Directory to output trained policies, logs, and plots. A subdirectory is created for each job. This is speficified relative to working directory'))
    parser.add_argument('--use_gpu', action="store_true")
    args = parser.parse_args()

    # Create an output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create environment
    env = myenv.InvertedPendulumEnv(frame_skip=frame_skip)

    # Create an NN model
    model = make_model(env.observation_space.shape[0])
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(lr=lr)

    # Setup a scaling factor
    _dataset = collect_data(env, n_train, rollout_length)
    observations = _dataset[0]
    scale = 1.0 / np.std(observations, axis=0)
    scale = scale.reshape((1, -1))
    scale = 1.0
    print("scale factor is {}.".format(scale))

    dataset = ([], [], [])
    train_loss = []
    for epoch in range(n_epochs):
        # Add MPC results to the dataset
        dataset_additional = collect_data(env, n_train, rollout_length)
        dataset = tuple([d1 + d2 for d1, d2 in zip(dataset, dataset_additional)])

        # Convert the dataset to tensorflow style
        train_dataset = tuple([np.array(d, dtype=np.float64) for d in dataset])
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(n_train * rollout_length).batch(batchsize)

        # Train the entire dataset
        loss_total = 0.0
        for s, a, s_next in tqdm.tqdm(train_dataset):
            with tf.GradientTape() as tape:
                # Compute NN outputs
                x = tf.concat((s, a), axis=1)
                s_predict = model(x)
                loss = loss_object(s_next*scale, s_predict*scale)

                # Update the model
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                loss_total += float(loss)

        # Print training loss
        train_loss.append(loss_total / len(dataset[0]))
        print("epoch {:4d}: train loss = {:.5e}".format(epoch, train_loss[-1]))

        # Add MPC results to the dataset
        #dataset_additional = collect_data_with_mpc(env, model, n_train, rollout_length)
        #dataset = tuple([d1 + d2 for d1, d2 in zip(dataset, dataset_additional)])

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

    for _ in tqdm.tqdm(range(n_rollout)):
        obs = env.reset()

        for _ in range(rollout_length):
            action = env.action_space.sample()

            obs_next, _, _, _ = env.step(action)

            observations.append(obs)
            actions.append(action)
            observations_next.append(obs_next)

            obs = obs_next

    #observations      = np.array(observations, dtype=np.float64)
    #actions           = np.array(actions, dtype=np.float64)
    #observations_next = np.array(observations_next, dtype=np.float64)

    return observations, actions, observations_next


def collect_data_grid(env, n_rollout, rollout_length):
    observations = []
    actions = []
    observations_next = []

    for _ in tqdm.tqdm(range(n_rollout)):
        obs = env.reset()

        for _ in range(rollout_length):
            action = env.action_space.sample()

            obs_next, _, _, _ = env.step(action)

            observations.append(obs)
            actions.append(action)
            observations_next.append(obs_next)

            obs = obs_next

    #observations      = np.array(observations, dtype=np.float64)
    #actions           = np.array(actions, dtype=np.float64)
    #observations_next = np.array(observations_next, dtype=np.float64)

    return observations, actions, observations_next


def collect_data_with_mpc(env, model, n_rollout, rollout_length):
    observations = []
    actions = []
    observations_next = []

    n_trials = 200
    horizon = 20
    policy = PddmPolicy(model, n_trials, horizon)

    for _ in tqdm.tqdm(range(n_rollout)):
        obs = env.reset()
        policy.reset()

        for _ in range(rollout_length):
            action = policy(obs)

            obs_next, _, _, _ = env.step(action)

            observations.append(obs)
            actions.append(action)
            observations_next.append(obs_next)

            obs = obs_next

    #observations      = np.array(observations, dtype=np.float64)
    #actions           = np.array(actions, dtype=np.float64)
    #observations_next = np.array(observations_next, dtype=np.float64)

    return observations, actions, observations_next


if __name__ == '__main__':
    tf.contrib.eager.enable_eager_execution()
    train()
