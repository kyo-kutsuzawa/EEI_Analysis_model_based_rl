import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

GYM_ASSET_PATH=xml_path = os.path.join(os.path.dirname(__file__), 'assets')


class SphereEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,file_path=os.path.join(GYM_ASSET_PATH, 'sphere1.xml')):
        mujoco_env.MujocoEnv.__init__(self, os.path.join(file_path), 4)
        utils.EzPickle.__init__(self)
        self.skip = self.frame_skip
    
    def step(self, a):
    	
    	# Only the control or power part of cost is implimented here
		# Cost corresponding to deviation from goal position written in agent 
		# since simulation needs no access to goal position

        ctrl_cost_coeff = 0.01
        reward_ctrl = -np.square(a).sum()
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, reward_ctrl, False, dict(reward_ctrl=reward_ctrl)
    
    def _get_obs(self):
    	pos = self.sim.data.qpos.flat[:2]
    	vel = self.sim.data.qvel.flat[:2]
    	#print type(pos), pos
    	#print type(vel), vel
    	obs = np.concatenate((pos,vel))
    	return obs
    """
    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        #print "*** qpos: ", qpos
        #print "*** qvel: ", qvel
        self.set_state(qpos, qvel)
        return self._get_obs()
        """

    def reset_model(self):
        self.num_step=0

        # set reset pose/vel
        self.reset_pose = self.init_qpos + self.np_random.uniform(
                        low=-.01, high=.01, size=self.model.nq)
        self.reset_vel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)

        #reset the env to that pose/vel
        return self.do_reset(self.reset_pose.copy(), self.reset_vel.copy())

        # Function to apply an external force to the joint of the pendulum.

    def do_reset(self, reset_pose, reset_vel, reset_goal=None):

        #reset
        self.set_state(reset_pose, reset_vel)

        #return
        return self._get_obs()
