import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import argparse
import myenv
from train import frame_skip, make_model, MpcPolicy, PddmPolicy


def evaluate():
    n_trials = 2000
    horizon = 10

    env = myenv.InvertedPendulumEnv(frame_skip=1)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = make_model(obs_dim)
    x = np.zeros((n_trials, obs_dim+act_dim))
    model(x)
    model.load_weights("result/param.h5")

    policy = PddmPolicy(model, n_trials, horizon)
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


def evaluate_eei():
    import matplotlib.pyplot as plt

    n_trials = 10
    length = 30
    i_disturb = 5
    disturb_interval = 1

    env = myenv.InvertedPendulumEnv(frame_skip=1)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = make_model(obs_dim)
    x = np.zeros((n_trials, obs_dim+act_dim))
    model(x)
    model.load_weights("result/param.h5")

    def trial(policy):
        eei_list = []
        c_pfm_list = []
        c_eng_list = []
        traj = []

        for n in range(n_trials):
            print("trial #{}".format(n))
            obs = env.reset()
            policy.reset()
            traj.append([])
            c_pfm = 0
            c_eng = 0
            for i in range(length):
                action = policy(obs)

                if i == i_disturb:
                    #env.add_disturb(np.random.uniform(0.1, 0.3))
                    env.add_disturb(0.2)
                elif i == i_disturb + disturb_interval:
                    env.add_disturb(0.0)

                for _ in range(frame_skip):
                    obs, _, _, _ = env.step(action)
                    traj[-1].append(np.concatenate((action, obs)))

                    if i >= i_disturb:
                        c_pfm += np.sum(obs[1]**2) * env.dt
                        c_eng += np.sum(np.abs(obs[3]*action)) * env.dt

            c_pfm_list.append(c_pfm)
            c_eng_list.append(c_eng)
            eei_list.append(1 / (c_pfm * c_eng))
        traj = np.array(traj)

        return eei_list, traj, c_pfm_list, c_eng_list

    print("evaluate mpc")
    n_samples = 2000
    horizon = 10
    policy = PddmPolicy(model, n_samples, horizon)
    eei_mpc, traj_mpc, c_pfm_mpc, c_eng_mpc = trial(policy)

    print("evaluate pd")
    policy = myenv.PdPolicy()
    policy.Kp = 7.0
    policy.Kp = 2.0
    policy.dt = env.dt
    eei_pd, traj_pd, c_pfm_pd, c_eng_pd = trial(policy)

    import seaborn as sns
    plt.figure()
    plt.suptitle("EEI")
    ax = sns.boxplot(x=["MPC", "PD"], y=[eei_mpc, eei_pd])
    ax = sns.swarmplot(x=["MPC"]*len(eei_mpc)+["PD"]*len(eei_pd), y=eei_mpc+eei_pd, color='black')
    plt.figure()
    plt.suptitle("Control deviation")
    ax = sns.boxplot(x=["MPC", "PD"], y=[c_pfm_mpc, c_pfm_pd])
    ax = sns.swarmplot(x=["MPC"]*len(c_pfm_mpc)+["PD"]*len(c_pfm_pd), y=c_pfm_mpc+c_pfm_pd, color='black')
    plt.figure()
    plt.suptitle("Energy consumption")
    ax = sns.boxplot(x=["MPC", "PD"], y=[c_eng_mpc, c_eng_pd])
    ax = sns.swarmplot(x=["MPC"]*len(c_eng_mpc)+["PD"]*len(c_eng_pd), y=c_eng_mpc+c_eng_pd, color='black')

    fig = plt.figure(figsize=(16, 8))
    n_col = np.ceil(np.sqrt(obs_dim+act_dim))
    n_row = np.ceil((obs_dim+act_dim)/n_col)
    axes = [fig.add_subplot(n_row, n_col, i+1) for i in range(obs_dim+act_dim)]
    labels = ["Input torque [Nm]", "Motor angle [rad]", "Pendulum angle [rad]", "Motor ang. vel. [rad/s]", "Pendulum ang. vel. [rad/s]" , "Measured torque [Nm]"]
    for i, ax in enumerate(axes):
        for n in range(n_trials):
            ax.plot(np.arange(traj_pd.shape[1]),  traj_pd[n, :, i],  color="C{}".format(n), ls="--")
            ax.plot(np.arange(traj_mpc.shape[1]), traj_mpc[n, :, i], color="C{}".format(n))
            ax.set_ylabel(labels[i])
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    tf.contrib.eager.enable_eager_execution()
    #evaluate_prediction()
    #evaluate()
    evaluate_eei()
