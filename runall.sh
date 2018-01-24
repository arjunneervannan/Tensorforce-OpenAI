#!/bin/bash
echo "Policy Gradient Methods"
# python openai_gym.py HalfCheetah-v1 -a trpo -lr 1 -e 40000 -m 400
# echo "done TRPO1"
# python openai_gym.py HalfCheetah-v1 -a vpg -lr 1 -e 40000 -m 400
# echo "done VPG1"
# python openai_gym.py HalfCheetah-v1 -a ppo -lr 1 -e 40000 -m 400
# echo "done PPO1"
# python openai_gym.py HalfCheetah-v1 -a trpo -lr 1e-1 -e 40000 -m 400
# echo "done TRPO1e-1"
# python openai_gym.py HalfCheetah-v1 -a vpg -lr 1e-1 -e 40000 -m 400
# echo "done VPG1e-1"
# python openai_gym.py HalfCheetah-v1 -a ppo -lr 1e-1 -e 40000 -m 400
# echo "done PPO1e-1"
# python openai_gym.py HalfCheetah-v1 -a trpo -lr 1e-2 -e 40000 -m 400
# echo "done TRPO1e-2"
# python openai_gym.py HalfCheetah-v1 -a vpg -lr 1e-2 -e 40000 -m 400
# echo "done VPG1e-2"
# python openai_gym.py HalfCheetah-v1 -a ppo -lr 1e-2 -e 40000 -m 400
# echo "done PPO1e-2"
# python openai_gym.py HalfCheetah-v1 -a trpo -lr 1e-3 -e 40000 -m 400
# echo "done TRPO1e-3"
# python openai_gym.py HalfCheetah-v1 -a vpg -lr 1e-3 -e 40000 -m 400
# echo "done VPG1e-3"
# python openai_gym.py HalfCheetah-v1 -a ppo -lr 1e-3 -e 40000 -m 400
# echo "done PPO1e-3"
# python openai_gym.py HalfCheetah-v1 -a trpo -lr 3e-4 -e 40000 -m 400
# echo "done TRPO3e-4"
# python openai_gym.py HalfCheetah-v1 -a vpg -lr 3e-4 -e 40000 -m 400
# echo "done VPG3e-4"
# python openai_gym.py HalfCheetah-v1 -a ppo -lr 3e-4 -e 40000 -m 400
# echo "done PPO3e-4"
# python openai_gym.py HalfCheetah-v1 -a trpo -lr 1e-4 -e 40000 -m 400
# echo "done TRPO1e-4"
# python openai_gym.py HalfCheetah-v1 -a vpg -lr 1e-4 -e 40000 -m 400
# echo "done VPG1e-4"
# python openai_gym.py HalfCheetah-v1 -a ppo -lr 1e-4 -e 40000 -m 400
# echo "done PPO1e-4"
python openai_gym.py HalfCheetah-v1 -a trpo -lr 1e-5 -e 40000 -m 400
echo "done TRPO1e-5"
python openai_gym.py HalfCheetah-v1 -a vpg -lr 1e-5 -e 40000 -m 400
echo "done VPG1e-5"
python openai_gym.py HalfCheetah-v1 -a ppo -lr 1e-5 -e 40000 -m 400
echo "done PPO1e-5"
