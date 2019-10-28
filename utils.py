import os, configargparse, time

import numpy as np
import pandas as pd

# import japanize_matplotlib

from scipy.stats import gaussian_kde, entropy, wasserstein_distance, multivariate_normal , zscore

def generate_params():
	parser = configargparse.ArgParser()
	parser.add('-dt', '--dt', dest='dt', type=float, default=0.1, help='Time of one frame')
	parser.add('-i', '--n_iterations', dest='n_iterations', type=int, default=10, help='Number of optimization iteration')
	parser.add('-s', '--select', dest='select', action='store_true', help='Flag of selecting time-window and team')
	parser.add('-l', '--load', dest='load', action='store_true', help='Flag of loading model')
	parser.add('-v', '--version', dest='version', type=int, help='Number of dataset version')
	parser.add('-n', '--name', dest='name', type=str, help='Name of Half')
	params_dict = vars(parser.parse_args())

	return params_dict

# def compute_emd(kde_list_list, range_dict):
def compute_emd(rv_list_list, range_dict):
	"""
	xx, yy = np.meshgrid(np.arange(range_dict['xmin'], range_dict['xmax'], 0.1), np.arange(range_dict['ymin'], range_dict['ymax'], 0.1))	
	positions = np.vstack([xx.ravel(), yy.ravel()])
	"""

	x, y = np.mgrid[range_dict['xmin']:range_dict['xmax']:.01, range_dict['ymin']:range_dict['ymax']:.01]
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x; pos[:, :, 1] = y

	emd_matrix = np.zeros((len(rv_list_list), len(rv_list_list)))
	for i, rv_list in enumerate(rv_list_list):
		Pn_list = [rv.pdf(pos) for rv in rv_list]

		for j, rv_list in enumerate(rv_list_list):
			if i >= j:
				continue

			Pn_list_tmp = [rv.pdf(pos) for rv in rv_list]
			emd_matrix[i, j] = emd_matrix[j, i] = wasserstein_distance(np.array(Pn_list).ravel(), np.array(Pn_list_tmp).ravel())

	return emd_matrix

def load_model(modeldir, n_roles=8):
	infile_list = os.listdir(modeldir)
	rv_dict = {}
	for infile in infile_list:
		with open(os.path.join(modeldir, infile), 'r') as fi:
			line_list = fi.readlines()
		rv_list = []
		for i in range(n_roles):
			st = i*6
			mean = np.array([float(m) for m in line_list[st+2].replace('\n','').split(' ')])
			cov = np.array([[float(v) for v in line_list[st+4+j].replace('\n','').split(' ')] for j in range(2)])
			rv_list.append(multivariate_normal(mean=mean, cov=cov))
		rv_dict[infile] = rv_list


	return rv_dict


def save_model(rv_dict, modeldir):
	for k, rv_list in rv_dict.items():
		with open(os.path.join(modeldir, k), 'w') as fo:
			for i, rv in enumerate(rv_list):
				fo.write(f'Role{i}\n')
				fo.write(f'pi:\n')
				for j, m in enumerate(rv.mean):
					if j == 0:
						fo.write(f'{m} ')
					else:
						fo.write(f'{m}\n')

				fo.write(f'cov:\n')
				for j, covs in enumerate(rv.cov):
					for k, cov in enumerate(covs):
						if k == 0:
							fo.write(f'{cov} ')
						else:
							fo.write(f'{cov}\n')

# split dataframe to {team}_{half}_{number}_{of or df}.csv
def split_dataframe(infile_path, dt=0.1, threshold_pass=0.5e+7, threshold_distance=2e+3):
	df = pd.read_csv(infile_path, header=None)

	df.columns = ['sid', 'ts', 'x', 'y', 'z', '|v|', '|a|', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
	left_r = 105
	right_r = 106

	ball_list = [4, 8, 10, 12]
	
	left_a_list = [13, 47, 49, 19, 53, 23, 57, 59]
	right_a_list = [14, 16, 88, 52, 54, 24, 58, 28]

	left_b_list = [61, 63, 65, 67, 69, 71, 73, 75]
	right_b_list = [62, 64, 66, 68, 38, 40, 74, 44]

	team_list = ['a', 'b']
	name_list = ['of', 'df']

	df['ts'] = [float(str(t)[:5]+'.'+str(t)[5:]) for t in df.ts.values.tolist()]
	st_1st = 10753.295594424116
	ed_1st = 12557.295594424116
	st_2nd = 13086.639146403495
	ed_2nd = 14879.639146403495

	df = df[(st_1st <= df.ts)&(df.ts < ed_2nd)]
	df.ts -= st_1st
	seg_dict = {'1st_1':[0, (ed_1st-st_1st)/2],'1st_2':[(ed_1st-st_1st)/2, ed_1st-st_1st] ,'2nd_1': [st_2nd-st_1st, (ed_2nd-st_2nd)/2-st_1st],'2nd_2':[(ed_2nd-st_2nd)/2-st_1st, ed_2nd-st_1st]}
	for key, seg in seg_dict.items():
		st_time = time.time()
		st, ed = seg
		st_tmp, ed_tmp = st, st+dt
		data_a_of_list, data_a_df_list, data_b_of_list, data_b_df_list = [], [], [], []
		while st_tmp < ed:
			ball_features = df[(st_tmp <= df.ts)&(df.ts < ed_tmp)&((df.sid == ball_list[0])|(df.sid == ball_list[1])|(df.sid == ball_list[2])|(df.sid == ball_list[3]))][['x', 'y','|v|']].mean().values
			d_array = np.array([[np.linalg.norm(df[(st_tmp <= df.ts)&(df.ts < ed_tmp)&((df.sid == left)|(df.sid == right))][['x', 'y']].mean().values-ball_features[:2]) for (left, right) in zip(left_list, right_list)] for (left_list, right_list) in zip([left_a_list, left_b_list], [right_a_list, right_b_list])])
			tid, pid = np.unravel_index(d_array.argmin(), d_array.shape)
			min_d = d_array.min()

			if ball_features[-1] > threshold_pass and st_tmp != st:
				pid = pid if min_d < threshold_distance and tid == pre_tid else pre_pid
				tid = pre_tid

			pre_tid, pre_pid = tid, pid
			data = np.array([[df[(st_tmp <= df.ts)&(df.ts < ed_tmp)&((df.sid == left)|(df.sid == right))][['x', 'y']].mean().values for (left, right) in zip(left_list, right_list)] for (left_list, right_list) in zip([left_a_list, left_b_list], [right_a_list, right_b_list])])
			if pid == 0:
				data_a_of_list.append(data[0].tolist())
				data_b_df_list.append(data[1].tolist())
			else:
				data_a_df_list.append(data[0].tolist())
				data_b_of_list.append(data[1].tolist())

			st_tmp += dt; ed_tmp += dt

		# save data_{}_{}_list to .csv
		for i, data_list_list in enumerate(zip([data_a_of_list, data_b_of_list], [data_a_df_list, data_b_df_list])):
			for j, data_list in enumerate(data_list_list):
				data_array = np.array(data_list)
				index_list = np.isnan(data_array.reshape(-1, 8*2)).any(axis=1)
				data_array = data_array[~index_list].reshape(-1, 8, 2)
				data_array = np.array(([zscore(data) for data in data_array]))

				np.savetxt(os.path.join('_csv', '{}_{}_{}.csv').format(team_list[i],key,name_list[j]), data_array.reshape(-1, 8*2), delimiter=',')
		print('{}: {}[sec]'.format(key, time.time()-st_time))

def main():
	infile_path = os.path.join('.', 'full-game')
	split_dataframe(infile_path)

if __name__ == '__main__':
	main()