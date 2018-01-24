# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
OpenAI gym execution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import time
import csv
import gym
import signal
import subprocess
import numpy as np
from gym.envs.mujoco import mujoco_env

from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
from tensorforce import TensorForceError


# python openai_gym_trpo.py HalfCheetah-v1 -a configs/trpo.json -n configs/mlp2_network.json -e 30000 -m 500
# python openai_gym_trpo.py HalfCheetah-v1 -a configs/dqn.json -n configs/mlp2_network.json -e 30000 -m 500
# python openai_gym_trpo.py HalfCheetah-v1 -a configs/vpg.json -n configs/mlp2_network.json -e 30000 -m 500
# python openai_gym_trpo.py HalfCheetah-v1 -a configs/ppo.json -n configs/mlp2_network.json -e 30000 -m 500
# python openai_gym_trpo.py HalfCheetah-v1 -a configs/nafjson -n configs/mlp2_network.json -e 30000 -m 500


# python openai_gym.py HalfCheetah-v1 -a ppo -lr 1 -e 10000 -m 500


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('gym_id', help="Id of the Gym environment")
    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
    parser.add_argument('-lr', '--learning-rate', type=float, default=None, help="Learning Rate")
    parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
    parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
    parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
    parser.add_argument('-l', '--load', help="Load agent from this dir")
    parser.add_argument('--monitor', help="Save results to this directory")
    parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
    parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
    parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")
    # above lines take arguments passed through terminal to define the network specs, agent used, etc.
    args = parser.parse_args()

    environment = OpenAIGym(
        gym_id=args.gym_id,
        monitor=args.monitor,
        monitor_safe=args.monitor_safe,
        monitor_video=args.monitor_video
    )


    # layer1 = (environment.states.get('shape')[0]) * 10
    # layer3 = (environment.actions.get('shape')[0]) * 10
    # layer2 = int(np.sqrt(layer1 * layer3))

    network_spec = [
        dict(type='dense', size=32, activation='tanh'),
        dict(type='dense', size=32, activation='tanh')
    ]
    
    if args.learning_rate is not None:
        learning_rate = args.learning_rate

    agent_type = args.agent_config
    print(agent_type)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    modelDir = os.path.join(os.getcwd(), agent_type+"model"+str(learning_rate))
    f = open(agent_type + "_results" + args.gym_id + "_" + str(learning_rate) + ".csv", 'wt')
    writer = csv.writer(f)
    
    # print(modelDir)
    # print(learning_rate)
    # agent_config = {u'step_optimizer': {u'learning_rate': learning_rate, u'type': u'gradient_descent'}, u'baseline': None, u'entropy_regularization': 0.01, u'batch_size': 4000, u'gae_lambda': None, u'likelihood_ratio_clipping': 0.2, u'discount': 0.99, u'optimization_steps': 5, u'baseline_optimizer': None, u'baseline_mode': None, u'type': u'ppo_agent'}

    if agent_type == 'trpo':
        agent_config = {u'baseline_mode': None,
        u'baseline': None,
        u'learning_rate': learning_rate,
        u'entropy_regularization': None,
        u'batch_size': 4000,
        u'gae_lambda': None,
        u'discount': 0.99,
        u'likelihood_ratio_clipping': None,
        u'baseline_optimizer': None,
        u'type': u'trpo_agent'}
        print("TRPO")
    elif agent_type == 'ppo':
        agent_config = {u'step_optimizer': {u'learning_rate': learning_rate, u'type': u'adam'},
        u'baseline': None,
        u'entropy_regularization': None,
        u'batch_size': 4000,
        u'gae_lambda': None,
        u'likelihood_ratio_clipping': None,
        u'discount': 0.99, u'optimization_steps': 5,
        u'baseline_optimizer': None,
        u'baseline_mode': None,
        u'type': u'ppo_agent'}
        print("PPO")
    elif agent_type == 'vpg':
        agent_config = {u'optimizer': {u'learning_rate': learning_rate, u'type': u'adam'},
        u'baseline': None,
        u'entropy_regularization': None,
        u'batch_size': 4000,
        u'gae_lambda': None,
        u'discount': 0.99,
        u'baseline_optimizer': None,
        u'baseline_mode': None,
        u'type': u'vpg_agent'}
        print("VPG")
    else:
        raise TensorForceError("Correct Agent Configuration Not Provided")

# learning rate 9e-4/np.sqrt(layer2)

    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states_spec=environment.states,
            actions_spec=environment.actions,
            network_spec=network_spec
        )
    )
    if args.load:
        load_dir = os.path.dirname(args.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(args.load)

    if args.debug:
        print("-" * 16)
        print("Configuration:")
        print(agent_config)

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    if args.debug:  # TODO: Timestep-based reporting
        report_episodes = 1
    else:
        report_episodes = 100

    print("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

    def episode_finished(r):
        # writer.writerow writes the episode number and its reward to a .csv file
        writer.writerow((r.episode, r.episode_rewards[-1]))
        if r.episode % report_episodes == 0:
            steps_per_second = r.timestep / (time.time() - r.start_time)
            print("Finished episode {} after {} timesteps. Steps Per Second {}".format(
                r.agent.episode, r.episode_timestep, steps_per_second
            ))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            print("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
            print("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))

            print("Elapsed time in mins:", (time.time() - t0)/60)
            print("Estimated Time Remaining in mins:", (((time.time()-t0) / r.episode)/60) * (args.episodes - r.episode))

        # the below loop saves the model
        if r.episode % 100 == 0:
            agent.save_model(directory=os.path.join(modelDir, "agent"))
            print("model saved at episode", r.episode)
            print("##################################\n")

        # the below loop records the simulation every 1000 steps using recordscreen.py
        if r.episode < 1000:
            if round(np.cbrt(r.episode), 4) % 1 == 0:
            # if r.episode % 10 == 0:
                agent.save_model(directory=os.path.join(modelDir, "agent"))
                print("model saved at episode", r.episode)
                print("##################################\n")
                agent007.restore_model(directory=modelDir)
                env = gym.make(str(args.gym_id))
                # env.env.model.ncam = 1
                # env.env.model.distance = 7 * 1.0
                # env.env.model.lookat[2] += .8
                # env.env.model.elevation = -20
                s = env.reset()
                done = False
                pro = subprocess.Popen("python recordscreen.py " + agent_type + str(r.episode) + "_" + str(learning_rate) + ".mp4 "+ "-s 600x550 --crop-top=100 --crop-left=150 -r 60",
                                       stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
                # pro takes various arguments that specify the pixel size to record the screen
                for i in range(args.max_episode_timesteps):
                    env.render()
                    action = agent007.act(s)
                    s, r, done, _ = env.step(action)
                    time.sleep(0.025)
                os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
                os.chdir("/home/raj/Desktop/tensorforce-master/examples")
                print("episode has been recorded!")
        else:
            if r.episode % 1000 == 0:
                agent007.restore_model(directory=modelDir)
                env = gym.make(str(args.gym_id))
                s = env.reset()
                done = False
                pro = subprocess.Popen("python recordscreen.py " + agent_type + str(r.episode) + ".mp4 "+ "-s 600x550 --crop-top=100 --crop-left=150 -r 60",
                                       stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
                # pro takes various arguments that specify the pixel size to record the screen
                for i in range(args.max_episode_timesteps):
                    env.render()
                    action = agent007.act(s)
                    s, r, done, _ = env.step(action)
                    time.sleep(0.025)
                os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
                os.chdir("/home/raj/Desktop/tensorforce-master/examples")
                print("episode has been recorded!")

        return True
     
    # re-defines another agent to simulate the model
    agent007 = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states_spec=environment.states,
            actions_spec=environment.actions,
            network_spec=network_spec
        )
    )

    t0 = time.time()
    runner.run(
        timesteps=args.timesteps,
        episodes=args.episodes,
        max_episode_timesteps=args.max_episode_timesteps,
        deterministic=args.deterministic,
        episode_finished=episode_finished
    )

    f.close()
    print("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))


if __name__ == '__main__':
    main()
