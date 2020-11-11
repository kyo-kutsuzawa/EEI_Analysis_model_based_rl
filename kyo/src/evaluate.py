import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import myenv
from train import frame_skip, make_model, MpcPolicy


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

        for _ in range(frame_skip):
            env.render()
            obs, _, _, _ = env.step(action)


def evaluate_prediction():
    import matplotlib.pyplot as plt

    n_trials = 6
    horizon = 20

    env = myenv.InvertedPendulumEnv(frame_skip=frame_skip)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = make_model(obs_dim)
    x = np.zeros((1, obs_dim+act_dim))
    model(x)
    model.load_weights("result/param.h5")

    observations_true = []
    observations_est  = []
    for n in range(n_trials):
        observations_true.append([])
        observations_est.append([])

        obs = env.reset()
        obs_est = obs
        for _ in range(horizon):
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
        for n in range(n_trials):
            ax.plot(np.arange(observations_true.shape[1]), observations_true[n, :, i], color="C{}".format(n), ls="--")
            ax.plot(np.arange(observations_est.shape[1]),  observations_est[n, :, i],  color="C{}".format(n))

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    tf.contrib.eager.enable_eager_execution()
    evaluate_prediction()
    evaluate()
