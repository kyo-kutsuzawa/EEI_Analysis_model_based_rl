import gym
import time
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import os
from pddm.utils.helper_funcs import *
##env = gym.make('pddm_sphere-v0')
#env = gym.make('pddm_furuta_inverted_pendulum-v0')
env,_ = create_env('pddm_sphere-v0')
gym_env= env.env
class PD(object):
    def __init__(self, kp, kd):
	# Initialize PD control with kp and kd
        self.kp = kp
        self.kd = kd
        self.prev_time = 0
        self.time_step = []
        self.pos_step = []
    
    def ploting_data(self, time, pos):
        self.time_step.append(time)
        self.pos_step.append(pos)
        
    def force(self, error, vel, pos):
        self.cur_time = self.prev_time + 1 
        f = self.kp*error - self.kd*vel
        self.prev_time = self.cur_time

        self.ploting_data(self.cur_time, pos)
        return f

def normalize(val):
    return ((val-min_val)/(float(max_val-min_val)))

# Initializing PD controllers with gains
pd_agent_x = PD(1, 0.5)
pd_agent_y = PD(1, 0.75)

x_goal, y_goal = (-3, 7)

# min and max on differences for calculating force (i.e. actuator constraints)
min_val, max_val = (-3, 3)
action_x = 0.0
action_y = 0.0
x = []
y = []

outdir =  os.path.join(os.path.dirname(__file__),'utils_random_save')#'~/Documents/pddm-master/pddm/utils_random_save/'
#gym_env.monitor.start(outdir, force=True)
# first step is to reset environment (mujoco requirement, so do it!)
gym_env.reset()
# select initial position and velocity
qinit = np.array([3, -9])
vinit = np.array([0, 0])
#gym_env.set_state(qinit,vinit)
gym_env.env.do_reset(qinit,vinit)
# get initial observation, required for the loop below
obs = gym_env.env._get_obs()
for _ in range(200):
    #print("**********************")
    #gym_env.render()

    pos = obs[0:2]
    vel = obs[2:4]
 
    error_x = x_goal - pos[0]    
    action_x = pd_agent_x.force(error_x, vel[0], pos[0])

    error_y = y_goal - pos[1]    
    action_y = pd_agent_y.force(error_y, vel[1], pos[1])
    
    # actuator limits
    if(action_x > 3):
        action_x = 3
    elif (action_x < -3):
        action_x = -3
  
    if(action_y > 3):
        action_y = 3
    elif (action_y < -3):
        action_y = -3
    
    action = np.array([action_x, action_y])
    obs = gym_env.step(action)[0]
#gym_env.monitor.close()

time = np.array(pd_agent_x.time_step)
x = np.array(pd_agent_x.pos_step)
y = np.array(pd_agent_y.pos_step)
plt.plot(time, x)
plt.plot(time, y)
plt.annotate("kp=1, kd=5",
             xy=(50, 1.7),
             textcoords='offset points')
#plt.show()
plt.savefig(outdir+"/hello2.png")
plt.clf()
plt.plot(x,y)
#plt.show()#print("+++++++++++++++++++++++++++++++++++")
plt.savefig(outdir+"/hello3.png")
#print(x)
