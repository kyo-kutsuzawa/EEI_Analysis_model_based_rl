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

import numpy as np
import copy
import scipy.stats as stats
import matplotlib.pyplot as plt

#my imports
from pddm.samplers import trajectory_sampler
from pddm.utils.helper_funcs import do_groundtruth_rollout
from pddm.utils.helper_funcs import turn_acs_into_acsK
from pddm.utils.calculate_costs import calculate_costs
import pddm

##added by hamada
from pddm.utils.utils import state_prediction_inverted_pendulum
from pddm.utils.utils import eei_inverted_pendulum

class CEM(object):

    def __init__(self, env, dyn_models, reward_func, rand_policy, use_ground_truth_dynamics,
                 execute_sideRollouts, plot_sideRollouts, params,save_dir,iter,sim_ver,control_delta):

        ###########
        ## params
        ###########
        self.K = params.K
        self.horizon = params.horizon
        self.N = params.num_control_samples
        self.rand_policy = rand_policy
        self.use_ground_truth_dynamics = use_ground_truth_dynamics
        self.dyn_models = dyn_models
        self.execute_sideRollouts = execute_sideRollouts
        self.plot_sideRollouts = plot_sideRollouts
        self.reward_func = reward_func
        self.env = copy.deepcopy(env)

        #############
        ## params for CEM controller
        #############
        self.max_iters = params.cem_max_iters
        self.num_elites = params.cem_num_elites
        self.sol_dim = self.env.env.action_space.shape[0] * self.horizon
        self.ub = 1
        self.lb = -1
        self.epsilon = 0.001
        self.alpha = 0
        #aded by Hamada
        self._iter= iter
        self._save_dir = save_dir
        self._sim_ver = sim_ver
        self._control_delta = control_delta



    def get_action(self, step_number, curr_state_K, actions_taken_so_far,
                   starting_fullenvstate, evaluating, take_exploratory_actions,iter,rollout_num):
        """Select optimal action

        Args:
            curr_state_K:
                current "state" as known by the dynamics model
                actually a concatenation of (1) current obs, and (K-1) past obs
            step_number:
                which step number the rollout is currently on (used to calculate costs)
            actions_taken_so_far:
                used to restore state of the env to correct place,
                when using ground-truth dynamics
            starting_fullenvstate
                full state of env before this rollout, used for env resets (when using ground-truth dynamics)
            evaluating
                if True: default to not having any noise on the executing action
            take_exploratory_actions
                if True: select action based on disagreement of ensembles
                if False: (default) select action based on predicted costs

        Returns:
            best_action: optimal action to perform, according to this controller
            resulting_states_list: predicted results of executing the candidate action sequences
        """

        #initial mean and var of the sampling normal dist
        mean = np.zeros((self.sol_dim,))
        var = 5 * np.ones((self.sol_dim,))
        X = stats.truncnorm(
            self.lb, self.ub, loc=np.zeros_like(mean), scale=np.ones_like(mean))

        #stop if variance is very low, or if enough iters
        t = 0
        while ((t < self.max_iters) and (np.max(var) > self.epsilon)):

            #variables
            lb_dist = mean - self.lb
            ub_dist = self.ub - mean
            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            #get samples
            all_samples_orig = X.rvs(size=[self.N, self.sol_dim]) * np.sqrt(
                constrained_var) + mean  # [N, ac*h]
            all_samples = all_samples_orig.reshape(
                self.N, self.horizon, -1)  #interpret each row as a sequence of actions
            all_samples = np.clip(all_samples, -1, 1)

            ########################################################################
            ### make each action element be (past K actions) instead of just (curr action)
            ########################################################################

            #all_samples is [N, horizon, ac_dim]
            all_acs = turn_acs_into_acsK(actions_taken_so_far, all_samples,
                                         self.K, self.N, self.horizon)
            #all_acs should now be [N, horizon, K, ac_dim]

            ############################
            ### have model predict the result of executing those candidate action sequences
            ############################

            if self.use_ground_truth_dynamics:
                paths = trajectory_sampler.sample_paths_parallel(
                    self.N,
                    all_samples,
                    actions_taken_so_far,
                    starting_fullenvstate,
                    self.env,
                    suppress_print=True,
                )  #list of dicts, each w observations/actions/etc.

                #the taken number of paths is num_cpu*(floor(self.N/num_cpu))
                #rather than self.N, so update parameter accordingly
                self.N = len(paths)
                all_samples = all_samples[:self.N]

                resulting_states = [entry['observations'] for entry in paths]
                resulting_states = np.swapaxes(resulting_states, 0, 1)

                resulting_states_list = [resulting_states]
            else:
                resulting_states_list = self.dyn_models.do_forward_sim(
                    [curr_state_K, 0],
                    np.copy(all_acs))
                resulting_states_list = np.swapaxes(
                    resulting_states_list, 0,
                    1)  #this is now [ensSize, horizon+1, N, statesize]

            ############################
            ### evaluate the predicted trajectories
            ############################

            #calculate costs : [N,]
            costs, mean_costs, std_costs = calculate_costs(resulting_states_list, all_samples,
                                    self.reward_func, evaluating, take_exploratory_actions)

            #pick elites, and refit mean/var
            #Note: these are costs, so pick the lowest few to be elites
            indices = np.argsort(costs)
            elites = all_samples_orig[indices][:self.num_elites]
            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            #interpolate between old mean and new one
            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            #next iteration
            t += 1

        #return the best action
        best_score = np.min(costs)
        best_sequence = mean.reshape(
            self.horizon, -1)  #interpret the 'row' as a sequence of actions
        best_action = np.copy(best_sequence[0])  #(acDim,)

        #return the worst action
        best_sim_num = np.argmin(costs)
        worst_sim_num = np.argmax(costs)

        #########################################
        ### execute the candidate action sequences on the real dynamics
        ### instead of just on the model
        ### useful for debugging/analysis...
        #########################################

        if self.execute_sideRollouts:
            if (step_number % self.horizon) == 0 :
                state_prediction_inverted_pendulum(self.env,self.reward_func, resulting_states_list, all_samples, starting_fullenvstate,
                                                   actions_taken_so_far,
                                                   self._save_dir, self.plot_sideRollouts, best_sim_num, worst_sim_num,
                                                   iter, rollout_num, step_number)
                """
                if self._sim_ver == "inverted_pendulum":
                    eei_inverted_pendulum(self.env, self.reward_func, resulting_states_list, all_samples,
                                               starting_fullenvstate,
                                               actions_taken_so_far,
                                               self._save_dir, self.plot_sideRollouts, best_sim_num,
                                               worst_sim_num,
                                               iter, rollout_num, step_number)
                                            """


        return best_action, resulting_states_list
