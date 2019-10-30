import os, shutil, configargparse, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import japanize_matplotlib

from scipy.stats import gaussian_kde, entropy, wasserstein_distance, multivariate_normal
from munkres import Munkres
from progressbar import ProgressBar
from sklearn.cluster import AgglomerativeClustering

from utils import generate_params, compute_emd, load_model, save_model 
from output_func import draw_pitch, plot_formation_distribution,  plot_formation_transition, plot_mean_formation_distribution

class AutoFormationDetector(object):
	"""docstring for TrackingAnalysis
	This class for running AutoFormationDetector

	Args:
		params_dict(dict) : Dictionary of Parameters
	"""
	def __init__(self, params_dict):
		self.__dict__ = params_dict.copy()

		# define using directory
		indir = os.path.join('_csv',f'ver{self.version}')
		self.figdir = os.path.join('_fig',f'ver{self.version}')
		self.modeldir = os.path.join('_model',f'ver{self.version}')
		if not self.load:
			for _dir in [self.figdir, self.modeldir]:
				if os.path.exists(_dir):
					shutil.rmtree(_dir)
				os.mkdir(_dir)

		infile_list = [infile for infile in os.listdir(indir) if infile.endswith('.csv')]
		
		if self.select:
			infile_list_tmp = infile_list
			infile_list = []
			for infile in infile_list_tmp:
				if infile.startswith('a_1st'):
					infile_list.append(infile)
				elif infile.startswith('b_2nd'):
					infile_list.append(infile)

		if self.version == 2:
			infile_list_tmp = infile_list
			infile_list = []
			for infile in infile_list_tmp:
				if not infile in ['a_1st_1_of.csv', 'b_1st_1_df.csv', 'a_2nd_1_of.csv', 'b_2nd_1_df.csv', 'a_2nd_2_of.csv', 'b_2nd_2_df.csv']:
					infile_list.append(infile)

		# read data_array_list
		self.data_dict = {infile.replace('.csv',''): np.loadtxt(os.path.join(indir,infile), delimiter=',').reshape(-1,8,2) for infile in infile_list}

		# arrange attacking direction to upper
		for key, data_array in self.data_dict.items():
			if key.startswith('a_2nd'):
				self.data_dict[key] = -data_array
			elif key.startswith('b_1st'):
				self.data_dict[key] = -data_array

		# get range_dict
		self.range_dict = {'xmin':[], 'xmax':[], 'ymin':[], 'ymax':[]}
		for k, data_array in self.data_dict.items():
			xmin, ymin = data_array.reshape(-1, 2).min(axis=0)
			xmax, ymax = data_array.reshape(-1, 2).max(axis=0)

			self.range_dict['xmin'].append(xmin); self.range_dict['xmax'].append(xmax)
			self.range_dict['ymin'].append(ymin); self.range_dict['ymax'].append(ymax)

		self.range_dict = {k:np.min(v) if 'min' in k else np.max(v) for k, v in self.range_dict.items()}

		# define munk
		self.munk = Munkres()

		# define number of clusters
		self.n_clusters = 3

	def compute_entropy(self, data_array):
		"""
		this module to compute entropy

		Args:
			data_array(np.ndarray) : dataset of tracking data, shape=(T, n_players, 2)

		Returns:
			- list of scipy.stats.multivariate_normal object
			- mean value of each position's KL-Divergence
			- list of KL-Divergence
		"""

		x, y = np.mgrid[self.range_dict['xmin']:self.range_dict['xmax']:.01, self.range_dict['ymin']:self.range_dict['ymax']:.01]
		pos = np.empty(x.shape + (2,))
		pos[:, :, 0] = x; pos[:, :, 1] = y

		rv_list = [multivariate_normal(mean=np.mean(data, axis=0), cov=np.cov(data.T)) for data in data_array]
		Pn_list = [rv.pdf(pos) for rv in rv_list]

		P = np.mean(np.array(Pn_list), axis=0)

		Vn_list = []
			
		for i, Pn in enumerate(Pn_list):
			Vn = entropy(Pn, P)
			Vn[Vn == np.inf] = 0
			Vn_list.append(Vn)
		
		V = np.mean(Vn_list)

		return rv_list, V, Pn_list

	def optimize_role_distribution(self, data_array, key=None):
		"""
		this module to optimize role distributions

		Args:
			key : name of data
			data_array(list) : shape = (T, n_players, 2)

		Returns:
			- list of optimized multivaiate_normal objects (list)
		"""

		# initialize gaussian_kde by all time frames
		rv_list, V, _ = self.compute_entropy(data_array.transpose(1,0,2))
		V_pre = V

		if key:
			plot_formation_distribution(rv_list, os.path.join(self.figdir,key+'_init.png'), self.range_dict)

		# optimize algorithm
		V_list = []
		V_list.append(V)
		for _iter in range(self.n_iterations):
			roles_list = []
			start_time = time.time()
			p = ProgressBar(maxval=len(data_array)).start()
			for t, data in enumerate(data_array):
				Et = np.array([[-np.log(rv.pdf(loc)) if rv.pdf(loc) != 0 else np.inf for rv in rv_list] for loc in data])
				roles = [r_tuple[1] for r_tuple in self.munk.compute(Et)]
				roles_list.append(roles)
				p.update(t+1)
			p.finish()

			data_array = np.array([data[roles] for (data, roles) in zip(data_array, roles_list)])
			# kde_list, V, _ = self.compute_entropy(data_array.transpose(1,0,2))
			rv_list, V, _ = self.compute_entropy(data_array.transpose(1,0,2))
			print('diff of V at iteration {}: {} ({} [sec])'.format(_iter, V_pre - V, time.time()-start_time))
			
			if V_pre - V <= 0 or np.isnan(V):
				if _iter != 0:
					# kde_list = kde_list_pre
					rv_list = rv_list_pre
				break

			V_pre = V; V_list.append(V); rv_list_pre = rv_list

		if key:
			plot_formation_distribution(rv_list, os.path.join(self.figdir, key+'_opt.png'), self.range_dict)

			# save decrease of V each iterations
			plt.figure(figsize=(5, 3))
			plt.plot(V_list)
			plt.xlabel('number of iterations')
			plt.ylabel('V')
			plt.savefig(os.path.join(self.figdir, key+'_V.png'))
			plt.close()

		return rv_list

	def run(self):
		"""
		this module to estimating set of role distribution, 
		and clustering all set of role distribution to K
		"""


		# optimize role distribution for each data
		if self.load:
			print('loading role distribution ...')
			self.rv_dict = load_model(self.modeldir)
		else:
			print('optimizing role distribution ...')
			self.rv_dict = {}
			for n, (key, data_array) in enumerate(self.data_dict.items()):
				print(f'optimize -> {key} ({n+1}/{len(self.data_dict)})')
				rv_list = self.optimize_role_distribution(key, data_array)

				# self.kde_dict[key] = kde_list
				self.rv_dict[key] = rv_list

			save_model(self.rv_dict, self.modeldir)

		# agglomerative clustering based on EMD(=Wasserstein distance)
		print('Running Agglomerative Clustering of role_distribution')
		emd_matrix = compute_emd(list(self.rv_dict.values()), self.range_dict)
		agg = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='precomputed', linkage='average')
		k_array = agg.fit_predict(emd_matrix)

		self.k_list = []
		for i, key in enumerate(self.rv_dict.keys()):
			print(f'{key} = {k_array[i]}')
			self.k_list.append(k_array[i])

		print('Plotting Clustering Results')
		plot_mean_formation_distribution(self.k_list, self.rv_dict, os.path.join(self.figdir, 'formation_clustering_results'), self.range_dict)

def main():
	params_dict = generate_params()

	auto_formation_detector = AutoFormationDetector(params_dict)
	auto_formation_detector.run()

	# simulate_online(auto_formation_detector)

if __name__ == '__main__':
	main()