# Tensorforce-OpenAI

The code in this repository runs on the half-cheetah environment from openAI and is an adaptation of the sample code from tensorforce's repository (https://github.com/reinforceio/tensorforce/blob/master/examples/openai_gym.py). This code also writes the reward outputs to a CSV file and records the model every 1000 episodes.

Two separate programs, plot.py and average.py calculate and plot the average from the CSV files.

See the video of the progress here: https://youtu.be/VK2rXLoEtW8.

**Read the paper that I wrote on this research topic here:** https://www.bjmc.lu.lv/fileadmin/user_upload/lu_portal/projekti/bjmc/Contents/6_4_02_Neervannan.pdf

## How to Run

- Clone the repository
- Run the command with `./runall.sh` to run all of the agents with various learning rates (see my other repository for more).
- You can also run the command like so: `python openai_gym.py HalfCheetah-v1 -a vpg -lr 1e-1 -e 40000 -m 400`. This would run the code on the HalfCheetah environment with the VPG agent and LR 1e-1 for 40,000 episodes, where each episode has 400 timesteps before timeout.

## Dependencies

You will need:
- OpenAI Gym
- Mujoco-py
- Tensorforce
- Tensorflow
- Numpy
- Matplotlib

## Citations
@misc{schaarschmidt2017tensorforce,
    author = {Schaarschmidt, Michael and Kuhnle, Alexander and Fricke, Kai},
    title = {TensorForce: A TensorFlow library for applied reinforcement learning},
    howpublished={Web page},
    url = {https://github.com/reinforceio/tensorforce},
    year = {2017}
}
