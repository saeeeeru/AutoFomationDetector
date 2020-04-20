import os, configargparse, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# import japanize_matplotlib

from scipy.stats import gaussian_kde, entropy, wasserstein_distance, multivariate_normal , zscore

cmap = plt.get_cmap('tab10')
twitter_color = "#141d26"

def simulate_online(auto_formation_detector, indir=os.path.join('_csv','ver1'), tau=600):
	"""
	this function to simulate AutoFormationDetector online

	Args:
		auto_formation_detector (object) : object of automatic_formation_detector class.
		indir(os.path.join(...)) : path to input file's directory
		tau(int) : sampling rate[Hz]
	"""
	
	# read data_array_list
	print('Loading data_array ...')
	data_dict = {team: np.vstack([np.loadtxt(os.path.join(indir,f'{team}_{auto_formation_detector.name}_{i}.csv'), delimiter=',').reshape(-1,8,2) for i in range(1,3)]) for team in ['a', 'b']}

	# plot role distribution and predicted cluster each teams
	T = int(len(data_dict[list(data_dict.keys())[0]])/tau)
	cluster_dict = {key: [] for key in data_dict.keys()}

	for t in range(T):
		print(f't={t} ...')
		for key, data_array in data_dict.items():
			print(f'key={key} ...')

			rv_list = auto_formation_detector.optimize_role_distribution(data_array[t*tau:(t+1)*tau])
			emd_matrix = compute_emd(list(auto_formation_detector.rv_dict.values())+[rv_list], auto_formation_detector.range_dict)
			cluster_dict[key].append(auto_formation_detector.k_list[np.argmin(emd_matrix[-1, :-1])])

	plot_formation_transition(cluster_dict, T, os.path.join(auto_formation_detector.figdir, f'formation_transition_{auto_formation_detector.name}.png'))

def draw_pitch():
	"""
	this function to draw pitches(vertically)

	Returns:
		- fig
		- ax
	"""
#     fig, ax = plt.subplots(1, 1, figsize=(15, 13))
	fig, ax = plt.subplots(1, 1, figsize=(7, 10))

	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['left'].set_visible(False)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['bottom'].set_visible(False)

	ax.plot([0, -50], [33965, -33960], color='black')
	ax.plot([22560, -50], [-33968, -33960], color='black')
	ax.plot([29880, 22560], [-33968, -33968], color='black', linewidth=10)
	ax.plot([29880, 52489], [-33968, -33939], color='black')
	ax.plot([52489, 52477], [-33939, 33941], color='black')
	ax.plot([52477, 29898], [33941, 33941], color='black')
	ax.plot([29898, 22578], [33941, 33941], color='black', linewidth=10)
	ax.plot([22578, 0], [33941, 33965], color='black')
	
	ax.plot([0, 52477], [0, 0], color='black')
	centreCircle = plt.Circle((52477/2, 0), 7000, color="black",fill=False)
	ax.add_patch(centreCircle)
	
	return fig, ax

def draw_pitch_rot():
	"""
	this function to draw pitches(horizontally)
	
	Returns:
		- fig
		- ax

	"""
	fig, ax = plt.subplots(1, 1, figsize=(10, 7))

	plt.gca().spines['right'].set_visible(False)
	plt.gca().spines['left'].set_visible(False)
	plt.gca().spines['top'].set_visible(False)
	plt.gca().spines['bottom'].set_visible(False)

	ax.plot([33965, -33960], [0, 50], color='black')
	ax.plot([-33968, -33960], [-22560, 50], color='black')
	ax.plot([-33968, -33968], [-29880, -22560], color='black', linewidth=10)
	ax.plot([-33968, -33939], [-29880, -52489], color='black')
	ax.plot([-33939, 33941], [-52489, -52477], color='black')
	ax.plot([33941, 33941], [-52477, -29898], color='black')
	ax.plot([33941, 33941], [-29898, -22578], color='black', linewidth=10)
	ax.plot([33941, 33965], [-22578, 0], color='black')

	ax.plot([0, 0], [0, -52477], color='black')
	centreCircle = plt.Circle((0, -52477/2), 7000, color="black",fill=False)
	ax.add_patch(centreCircle)

	return fig, ax

def plot_formation_distribution(rv_list, fpath, range_dict, mesh_size, attacking_direction):
	"""
	this function to plot formation distribution

	Args:
		rv_list (list): list of each scipy.stats.multivariate_normal object
		fpath (os.path.join(...)): path to output directory
		range_dict (dict): dictionary of range parameters
		mesh_size (float): 
		attacking_direction (str) : in ['left', 'right']
	"""


	x, y = np.mgrid[range_dict['xmin']:range_dict['xmax']:mesh_size, range_dict['ymin']:range_dict['ymax']:mesh_size]
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x; pos[:, :, 1] = y

	Pn_list = [rv.pdf(pos) for rv in rv_list]
	P = np.mean(np.array(Pn_list), axis=0)

	ratio = (range_dict['ymax']-range_dict['ymin']) / (range_dict['xmax']-range_dict['xmin'])
	fig, ax = plt.subplots(1, 1, figsize=(10.5*1.5, 6.8*1.5), facecolor=twitter_color)

	# ax.set_xlim(range_dict['xmin'], range_dict['xmax']); ax.set_ylim(range_dict['ymin'], range_dict['ymax'])
		
	for i, (rv, Pn) in enumerate(zip(rv_list, Pn_list)):
		# add center
		x_center, y_center = rv.mean
		ax.scatter(x_center, y_center, s=200, marker='^', facecolors='none', edgecolors=cmap(i))
		
		# add density
		CS = ax.contour(x, y, Pn, levels=np.linspace(np.min(Pn),np.max(Pn),10)[-2:], colors=[cmap(i)])
		ax.clabel(CS, inline=1, fontsize=10)

	# set atacking direction
	if attacking_direction:
		ax.text(0, -1.35, 'attacking', va='bottom', ha='center', fontsize=15, color='w')

	if attacking_direction == 'right':
		ax.arrow(x=-0.25,y=-1.4,dx=0.5,dy=0,head_width=0.05,head_length=0.05,length_includes_head=True,color='w')
	elif attacking_direction == 'left':
		ax.arrow(x=0.25,y=-1.3,dx=-0.5,dy=0,head_width=0.05,head_length=0.05,length_includes_head=True,color='w')

	# remove axis labels and ticks
	ax.set_xticklabels([]); ax.set_yticklabels([])
	ax.set_xticks([]); ax.set_yticks([])
	ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
	ax.spines['top'].set_visible(False); ax.spines['bottom'].set_visible(False)
	
	ax.set_facecolor(twitter_color)

	ax.set_xlim([-1.5,1.5]); ax.set_ylim([-1.5,1.5])

	# set title
	ax.set_title(fpath.split('/')[-1].replace('.png',''), color='w')
	
	# save figure
	plt.savefig(fpath, bbox_inches='tight', facecolor=twitter_color)
	plt.close()

def plot_formation_transition(cluster_dict, T, fpath):
	"""
	this function to plot formation transition (segmentation bar)

	Args:
		cluster_dict(dict) : dictionary of each cluster's cutpoints
		T(int) : number of windows
		fpath(os.path.join(...)) : path to output dictionary

	"""
	fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 4))

	for i, (key, cluster_list) in enumerate(cluster_dict.items()):
		ax = axes[i]
		for t, k in enumerate(cluster_list):
			rect = patches.Rectangle((t, 0), width=1, height=1, color=cmap(k))
			ax.add_patch(rect)
		ax.get_yaxis().set_visible(False)
		ax.set_xlim([0, T]); ax.set_xlabel('Time [min]')
		ax.set_title(key)

	plt.savefig(fpath)
	plt.close()

def plot_mean_formation_distribution(k_list, rv_dict, fpath, range_dict):
	"""
	this function to plot mean formation distribution

	Args:
		k_list(list) : list of number of clusters
		rv_list(list) : list of each scipy.stats.multivariate_normal object
		fpath (ois.path.join(...)) : path to output directory
		range_dict(dict) : dictionary of range parameters
	"""
	n_clusters = len(np.unique(k_list))
	fig, axes = plt.subplots(nrows=1, ncols=n_clusters, figsize=(4*n_clusters, 4*n_clusters/1.91))
	for k, ax in enumerate(axes):
		ax.set_title(f'Cluster {k+1}: {100*(len(np.where(np.array(k_list) == k)[0])/len(k_list))}%')
		ax.hlines(y=0, xmin=range_dict['xmin'], xmax=range_dict['xmax'], colors='black', linewidth=0.5)
		ax.set_xlim(range_dict['xmin'], range_dict['xmax'])
		ax.set_ylim(range_dict['ymin'], range_dict['ymax'])
		ax.set_xticklabels([]); ax.set_yticklabels([])

	mean_dict = {f'Cluster {k+1}':[] for k in range(n_clusters)}
	for i, (key, rv_list) in enumerate(rv_dict.items()):
		k = k_list[i]
		ax = axes[k]

		mean_list = []
		if key.startswith('b'):
			rv_list = np.array(rv_list)[[0, 4, 5, 2, 1, 3, 7, 6]].tolist()

		for j, rv in enumerate(rv_list):
			x, y = rv.mean
			ax.scatter(x, y, s=50, marker='^', facecolors='none', edgecolors=cmap(j))
			mean_list.append([x, y])

		mean_dict[f'Cluster {k+1}'].append(mean_list)

	for k, mean_list in enumerate(mean_dict.values()):
		mean_array = np.mean(np.array(mean_list), axis=0)
		for j, mean in enumerate(mean_array):
			axes[k].scatter(mean[0], mean[1], s=75, c=cmap(j))

	# plt.savefig(fpath, bbox_inches="tight")
	plt.savefig(fpath)
	plt.close()
