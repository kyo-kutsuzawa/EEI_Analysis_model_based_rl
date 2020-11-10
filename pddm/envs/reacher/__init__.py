from gym.envs.registration import register

register(
    id='pddm_reacher-v0',
    entry_point='pddm.envs.reacher.reacher:ReacherEnv',
    max_episode_steps=1000,
)

