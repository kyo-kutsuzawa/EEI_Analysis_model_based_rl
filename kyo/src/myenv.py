import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, frame_skip=1):
        utils.EzPickle.__init__(self)

        self.num_step = 0  # discrete time index

        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_ip.xml")
        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip)

        self.frame_skip = frame_skip

    def step(self, action):
        # Step the simulation
        self.num_step += 1
        self.do_simulation(action, self.frame_skip)

        # Get the current observation
        obs = self._get_obs()
        pendulum_angle = obs[1]

        # Compute reward
        action = np.array(action)
        r_action = -0.1 * np.sum(action**2)
        r_stable = 10 - 50 * np.abs(pendulum_angle)
        reward = r_stable + r_action

        # Check the terminal condition
        if np.abs(pendulum_angle) > 360:
            done = True
        else:
            done = False

        return obs, reward, done, None

    def _get_obs(self):
        joints_pos = self.sim.data.qpos.flatten()
        joints_vel = self.sim.data.qvel.flatten()

        return np.concatenate([joints_pos, joints_vel])

    def reset_model(self):
        # Reset time
        self.num_step = 0

        # Reset joint state
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(qpos, qvel)

        # first observation
        obs = self._get_obs()

        return obs

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -10
        self.viewer.cam.azimuth = 180


if __name__ == "__main__":
    env = InvertedPendulumEnv(frame_skip=5)
    env.reset()

    while True:
        env.render()

        action = env.action_space.sample()
        obs, _, _, _ = env.step(action)

        for o in obs:
            print("{:8.3f}".format(o), end="  ")
        print("", end="\r")
