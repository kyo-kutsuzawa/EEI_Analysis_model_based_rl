from gym.envs.registration import register

register(
    id='pddm_sphere-v0',
    entry_point='pddm.envs.sphere.sphere:SphereEnv',
    max_episode_steps=1000,
)