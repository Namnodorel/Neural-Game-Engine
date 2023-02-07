from scs_architecture.moonlander_environment.envs import MoonlanderWorldEnv

from training.environment.data_generator import DataGenerator

import random as rnd
import numpy as np


class MoonlanderDataGenerator(DataGenerator):

    def __init__(self):
        super().__init__('MoonlanderGenerator')
        self.env = MoonlanderWorldEnv(
            world_x_width=40,
            world_y_height=500,
            size=2,
            level_difficulty="hard",
            agent_observation_height=10,
            verbose_level=0,
            drift_length=20,
            use_variable_drift_intensity=False,
            invisible_drift_probability=0.25,
            object_type="obstacle"
        )

    def get_generator_params(self):
        return {
            # 'num_actions': self.get_num_actions()
        }

    def get_num_actions(self):
        return 3

    def get_tile_size(self):
        return 1

    def generate_samples(self, batch_size):
        """
        Batches must be generated in the form (batch, channels, width, height)
        :param batch_size:
        :return:
        """
        self._logger.debug(f'Generating 1 batch with batch size: {batch_size}')

        self.env.reset()
        # Attempt to generate samples from more interesting places than just the start
        self.env.randomize_position()

        observations = []
        actions = []
        rewards = []

        for b in range(batch_size + 1):
            action = rnd.randint(0, 2)
            observation, reward, done, _ = self.env.step(action)

            # Normalize RGB values to [0, 1]
            observation = observation / 255.0
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            if done:
                self.env.reset()
                # self.env.randomize_position()


        np_observations = np.stack(observations)
        np_rewards = np.stack(rewards)

        input_observation_batch = np_observations[:-1]
        expected_observation_batch = np_observations[1:]
        expected_reward_batch = np_rewards[1:]
        input_action_batch = np.stack(actions)[1:]

        return [
            {
                'input_observation_batch': input_observation_batch,
                'input_action_batch': input_action_batch,
                'expected_observation_batch': expected_observation_batch,
                'expected_reward_batch': expected_reward_batch,
            }
        ]

    def get_data_shape(self):
        raise NotImplementedError()
