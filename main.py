# -*- coding: utf-8 -*-

"""
Created on Fri Dec  9 07:43:14 2022

@author: ALidtke
"""

from numpy import zeros, ones, array, NINF
import pandas
from torch import nn

from matplotlib import rc, rcParams
import matplotlib.pyplot as plt

import stable_baselines3
import sb3_contrib
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise

import os, sys, shutil, yaml

import verySimpleAuv.verySimpleAuv as auv
from verySimpleAuv import resources

font = {"family": "serif", "weight": "normal", "size": 16}

rc("font", **font)
rcParams["figure.figsize"] = (6, 2.7)

if __name__ == "__main__":
	
	# USED FOR FOLDER NAMES
	agent_name = "tmp_agent"
	
	# SCRIPT CONFIG
	do_training = False
	do_evaluation = True
	
	i_agent2evaluate = 2 # if do_training is false, choose agent index to evaluate
	
	# TRAINING CONFIG
	n_process = 12
	n_agent = 3
	overwrite = True # if True, will remove the folder with the same agent name if exists
	n_train_step = 1000000 # 1_500_000
	
	agent_kwargs = {
		'verbose': 1,
		'gamma': 0.95,
		'batch_size': 256,
		"gradient_steps": 1,
		'buffer_size': (128*3)*512,
		'learning_rate': 2e-3,
		'learning_starts': 256,
		'train_freq': (1, "step"),
		"action_noise": VectorizedActionNoise(NormalActionNoise(
			zeros(3), 0.05*ones(3)), n_process),
	}
	policy_kwargs = {
		"activation_fn": nn.GELU,
		"net_arch": dict(
			pi=[32, 32, 32],
			qf=[32, 32, 32],
		),
	}
	env_kwargs = {
		"currentVelScale": 1.,
		"currentTurbScale": 2.,
		"noiseMagActuation": 0.1,
		"noiseMagCoeffs": 0.1,
	}

	# EVAL CONFIG
	makeAnimation = True
	
	env_kwargs_evaluation = {
		"currentVelScale": 1.,
		"currentTurbScale": 2.,
		"noiseMagActuation": 0.,
		"noiseMagCoeffs": 0.,
	}
	
	env_kwargs_evaluation_noise = {
		"currentVelScale": 1.,
		"currentTurbScale": 2.,
		"noiseMagActuation": 0.1,
		"noiseMagCoeffs": 0.1,
	}

	eval_seed = 1

	# TRAINING
	log_folder = f"./agents/{agent_name}"
	log_folder = os.path.abspath(log_folder) + "/" # verySimpleAuv needs absolute path, dirty fix
	if do_training:
		# Train multiple RL agents with identical configurations to identify variability due to luck
		training_rewards = []
		training_times = []
		agents = []
		
		try:
			os.makedirs(log_folder)
		except FileExistsError as _:
			if overwrite:
				shutil.rmtree(log_folder)
			else:
				print("[ERROR] The agent directory already exists and overwrite=False!")
				sys.exit(1)
		
		for i_agent in range(n_agent):
			
			agent_folder = log_folder + f"agent_{i_agent}/"
			os.makedirs(agent_folder)
			
			# Env
			env = SubprocVecEnv([auv.make_env(i, env_kwargs=env_kwargs) for i in range(n_process)])
			env = VecMonitor(env, agent_folder)
			
			# Agent
			# agent = stable_baselines3.DDPG("MlpPolicy", env, policy_kwargs=policy_kwargs, **agent_kwargs)
			agent = sb3_contrib.TQC("MlpPolicy", env, policy_kwargs=policy_kwargs, **agent_kwargs)
			# agent = sb3_contrib.RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs, **agent_kwargs)
			# agent = stable_baselines3.SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, **agent_kwargs)
			
			# Training
			rewards, training_time = resources.trainAgent(agent, n_train_step, agent_folder)
			training_rewards.append(rewards)
			training_times.append(training_time)
			agents.append(agent)
			
			# Save the model and replay buffer
			agent.save(agent_folder + "model")
			
			try:
				# Models like PPO don't have it
				agent.save_replay_buffer(agent_folder + "replay_buffer")
			except AttributeError:
				pass
			
			# Save hyperparameters
			resources.saveHyperparameteres(
				agent_name, agent_kwargs, policy_kwargs, env_kwargs, n_train_step, training_times, n_process, agent_folder + "hyperparameters.yaml")
				
		# finding the "best" agent
		best_reward = NINF
		for i_agent, curr_reward in enumerate(training_rewards):
			roll_len = 50 if len(curr_reward["r"]) > 500 else int(len(curr_reward["r"])/10) + 1 # kinda dirty but oh well
			
			rol = curr_reward.rolling(200).mean()
			if rol["r"].values[-1] > best_reward:
				i_best = i_agent
				best_reward = rol["r"].values[-1]
		
		# Evaluation env for a quick check
		env_eval = auv.AuvEnv(**env_kwargs_evaluation)
		
		# Trained agent
		print("\nAfter training")
		mean_reward, median_reward, all_rewards = resources.evaluate_agent(agent, env_eval)
		resources.plotEpisode(env_eval, "RL control")
		
		# PD controller benchmark
		print("\nSimple control")
		env_eval_pd = auv.AuvEnv()
		pdController = auv.PDController(env_eval_pd.dt)
		mean_reward, median_reward, all_rewards = resources.evaluate_agent(pdController, env_eval_pd)
		fig, ax = resources.plotEpisode(env_eval_pd, "PD Controller")
	
	# EVALUATION
	if do_evaluation:
		if not do_training:
			i_best = i_agent2evaluate
			agent_folder = log_folder + f"agent_{i_best}/"
		
		best_agent = agent_folder + f"model"
		
		# Env and agent
		env_eval = auv.AuvEnv(**env_kwargs_evaluation,seed=eval_seed)
		best_agent = sb3_contrib.TQC.load(best_agent)
		# best_agent = stable_baselines3.DDPG.load(best_agent)

		# Load the hyperparamters
		with open(agent_folder + "hyperparameters.yaml", "r") as file:
			hyperparameters = yaml.safe_load(file)

		# Agent reward during training 
		reward_history = pandas.read_csv(agent_folder + "monitor.csv", skiprows=1)
		
		fig, ax = plt.subplots()
		ax.set_xlabel("Episode")
		ax.set_ylabel("Reward")
		
		roll_len = 50 if len(reward_history["r"]) > 500 else int(len(reward_history["r"])/10) + 1 # kinda dirty but oh well
		
		ax.scatter(reward_history.index, reward_history["r"], marker="x", zorder=10)
		ax.plot(reward_history.index, reward_history.rolling(roll_len).mean()["r"], c="r", lw=2)
		
		plt.savefig("./rewards.png",bbox_inches='tight', pad_inches=0.2)
		
		# Evaluate for a large number of episodes to test robustness.
		print("\nRL agent")
		mean_reward, median_reward, allRewards = resources.evaluate_agent(
			 best_agent, env_eval, eval_seed, num_episodes=10)
		
		# PD Controller
		print("\nSimple control")
		env_eval_pd = auv.AuvEnv(**env_kwargs_evaluation,seed=eval_seed)
		pdController = auv.PDController(env_eval_pd.dt)
		# mean_reward_pd, median_reward_pd, allRewards_pd = resources.evaluate_agent(
		# 	 pdController, env_eval_pd, num_episodes=10)

		mean_reward_pd, median_reward_pd, allRewards_pd = resources.evaluate_agent(
			 pdController, env_eval, num_episodes=10,seedy=eval_seed)
		
		# Evaluate once with fixed initial conditions.
		print("\nLike-for-like comparison")
		print("RL agent")
		resources.evaluate_agent(best_agent, env_eval, num_episodes=1, init=[[-0.5, -0.5], 0.785, 1.57],seedy=eval_seed)
		
		print("PD controller")
		resources.evaluate_agent(pdController, env_eval_pd, num_episodes=1, init=[[-0.5, -0.5], 0.785, 1.57],seedy=eval_seed)
		resources.plotEpisode(env_eval, "RL control fixed init")
		plt.savefig("./rl_episode.png",bbox_inches='tight', pad_inches=0.2)
		#resources.plotEpisode(env_eval_pd, "Simple control fixed init")
		resources.plotEpisode(env_eval, "Simple control fixed init")
		
		plt.savefig("./pd_episode.png",bbox_inches='tight', pad_inches=0.2)

		# Compare stats
		fig, ax = plt.subplots()
		ax.set_xlabel("Episode")
		ax.set_ylabel("Reward")
		x = array(range(len(allRewards)))
		ax.bar(x-0.4, allRewards, 0.4, align="edge", color="r", label="RL control")
		ax.bar(x, allRewards_pd, 0.4, align="edge", color="b", label="Simple control")
		xlim = ax.get_xlim()
		ax.plot(xlim, [mean_reward]*2, "r--", lw=4, alpha=0.5)
		ax.plot(xlim, [mean_reward_pd]*2, "b--", lw=4, alpha=0.5)
		ax.plot(xlim, [0]*2, "k-", lw=1)
		ax.set_xlim(xlim)
		ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
		
		plt.tight_layout()
		plt.savefig("./compare.png")

		# Animate
		if makeAnimation:
			resources.animateEpisode(env_eval, "RL_control",
								  flipX=True, Uinf=env_kwargs_evaluation["currentVelScale"])
			resources.animateEpisode(env_eval_pd, "naive_control",
								  flipX=True, Uinf=env_kwargs_evaluation["currentVelScale"])

