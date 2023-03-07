import matplotlib.pyplot as plt
import numpy as np
from helper_functions import compute_mse
import pandas as pd
from scipy.fft import fft, fftfreq
from matplotlib.patches import Rectangle
from sklearn.metrics import r2_score
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib as mpl
from string import ascii_lowercase

FIGSIZE = (30, 21)
COLOR_1 = 'darkgreen'
COLOR_2 = 'indigo'
BIG_FONT_SIZE = 66
SMALL_FONT_SIZE = 62
mpl.rcParams.update({'font.size': SMALL_FONT_SIZE,
					 'axes.titlesize': BIG_FONT_SIZE,
					 'figure.figsize': FIGSIZE,
					 'legend.fontsize': SMALL_FONT_SIZE,
					 'xtick.labelsize': SMALL_FONT_SIZE,
					 'ytick.labelsize': SMALL_FONT_SIZE,
                     'lines.linewidth': 4,
					 'figure.autolayout': True})

def plot_coeffs(model, key, op_idx, coeff_axs):
	# Plot barchart of coefficients for different terms for BlPitch1 and P_motor
	op_label = key
	candidate_funcs = model.get_feature_names()
	coeffs = model.coefficients()[op_idx, :]
	sorted_coeff_indices = np.flip(np.argsort(np.abs(coeffs)))
	candidate_funcs = np.array(candidate_funcs)[sorted_coeff_indices]
	coeffs = np.array(coeffs)[sorted_coeff_indices]

	if hasattr(coeff_axs, 'shape'):
		ax = coeff_axs[op_idx]
	else:
		ax = coeff_axs

	ax.bar(range(1, len(candidate_funcs) + 1), coeffs)
	ax.set(xlabel='Feature Index', ylabel='Coefficient Value', title=op_label)

	coeffs_df = pd.DataFrame(
		data={'Candidate Function': candidate_funcs, 'Coefficient': coeffs})

	print("SINDY MODEL COEFFICIENTS")
	print(f"\n{coeffs_df}")
	# print(f'\n{op_label} Coefficients in order of absolute value:')
	# for ii in range(len(coeffs)):
	#     print(f'{candidate_funcs[ii]} - {coeffs[ii]}')
	return coeffs_df


def plot_ensemble_error(model, state_cols, op_idx, ensemble_axs):
	median_ensemble = np.median(model.coef_list, axis=0)

	# mean_ensemble = np.mean(model.coef_list, axis=0)
	std_ensemble = np.std(model.coef_list, axis=0)
	mean_ensemble = np.mean(model.coef_list, axis=0)
	norm_var_ensemble = std_ensemble ** 2 / abs(mean_ensemble)
	candidate_funcs = model.get_feature_names()

	# xticknames = [f'${f}$' for f in model.get_feature_names()]

	sorted_coeff_indices = np.flip(np.argsort(np.abs(median_ensemble[op_idx, :])))
	mean_ensemble = np.array(mean_ensemble[op_idx, :])[sorted_coeff_indices]
	median_ensemble = np.array(median_ensemble[op_idx, :])[sorted_coeff_indices]
	std_ensemble = np.array(std_ensemble[op_idx, :])[sorted_coeff_indices]
	norm_var_ensemble = np.array(norm_var_ensemble[op_idx, :])[sorted_coeff_indices]
	candidate_funcs = np.array(candidate_funcs)[sorted_coeff_indices]
	coeffs_df = pd.DataFrame(
		data={'Candidate Function': candidate_funcs,
			  # 'Mean': mean_ensemble,
			  'Median': median_ensemble,
			  'Normalized Variance': norm_var_ensemble
			  })

	print("\nDATASET ENSEMBLE COEFFICIENT ERROR")
	print(f"\n{coeffs_df}")

	# for i in range(len(state_cols[key])):
	if hasattr(ensemble_axs, 'shape'):
		ax = ensemble_axs[op_idx]
	else:
		ax = ensemble_axs

	ax.errorbar(range(1, len(model.get_feature_names()) + 1), mean_ensemble,
				yerr=std_ensemble,
				fmt='o', color=COLOR_1)
	ax.set(title=f'Median Ensembling for {state_cols[op_idx]}', xlabel='Feature Index')
	return coeffs_df


def sum_contributions(transform_iter, n_output_features, coeffs, x, u, t):
	# contribs = np.vstack([contribs, np.zeros(model.n_output_features_)])
	# contribs = np.zeros(model.n_output_features_)
	print(F'Summing contributions...')
	contribs = np.zeros(n_output_features)
	for k in range(len(t)):
		xk = x[k, :]
		uk = u[k, :]
		Xt = np.concatenate((xk, uk))[np.newaxis, :]

		# for _, name, transform in model.model._iter(with_final=False):
		for transform in transform_iter:
			Xt = transform(Xt)

		# contribs += (Xt * model.model.steps[-1][1].optimizer.coef_)[0]
		contribs += (Xt * coeffs)[0]
	# + models[key].model.steps[-1][1].optimizer.intercept_
	return contribs


def plot_contribution_error(model, key, op_idx, contribution_axs, data_processor, test_on_training_data, parallel):
	if test_on_training_data:
		t = data_processor.t_train
		x = data_processor.x_train[key]
		u = data_processor.u_train[key]
		n_datasets = data_processor.n_training_datasets
	else:
		t = data_processor.t_test
		x = data_processor.x_test[key]
		u = data_processor.u_test[key]
		n_datasets = data_processor.n_testing_datasets

	# contribs = np.zeros((0, model.n_output_features_))
	if parallel:
		pool = Pool(mp.cpu_count())
		contribs = np.vstack(pool.starmap(sum_contributions, [([item[-1].transform for item in model.model._iter(with_final=False)],
															   model.n_output_features_,
															   model.model.steps[-1][1].optimizer.coef_,
															   x[d], u[d], t) for d in range(n_datasets)]))
		pool.close()
	else:
		contribs = []
		for d in range(n_datasets):
			contribs.append(sum_contributions([item[-1].transform for item in model.model._iter(with_final=False)],
											  model.n_output_features_,
											  model.model.steps[-1][1].optimizer.coef_,
											  x[d], u[d], t))
		contribs = np.array(contribs)

	# average contribution of each output feature over full time series, with dimensions (n_datasets, n_output_features)
	contribs /= len(t)

	# get standard deviation over multiple datasets
	std_contrib = np.std(contribs, axis=0)

	# get mean over multiple datasets
	mean_contrib = np.mean(contribs, axis=0)
	median_contrib = np.median(contribs, axis=0)
	norm_var_contrib = std_contrib ** 2 / abs(mean_contrib)
	candidate_funcs = model.get_feature_names()

	sorted_contrib_indices = np.flip(np.argsort(np.abs(mean_contrib)))
	mean_contrib = np.array(mean_contrib)[sorted_contrib_indices]
	median_contrib = np.array(median_contrib)[sorted_contrib_indices]
	std_contrib = np.array(std_contrib)[sorted_contrib_indices]
	norm_var_contrib = np.array(norm_var_contrib)[sorted_contrib_indices]
	candidate_funcs = np.array(candidate_funcs)[sorted_contrib_indices]
	contrib_df = pd.DataFrame(
		data={'Candidate Function': candidate_funcs,
			  'Median': median_contrib,
			  'Normalized Variance': norm_var_contrib})

	print("CANDIDATE FUNCTION CONTRIBUTIONS")
	print(f"\n{contrib_df}")

	# for i in range(len(state_cols[key])):
	if hasattr(contribution_axs, 'shape'):
		ax = contribution_axs[op_idx]
	else:
		ax = contribution_axs

	ax.errorbar(range(1, len(model.get_feature_names()) + 1), mean_contrib,
				yerr=std_contrib,
				fmt='o', color=COLOR_1)
	ax.set(title=f'Relative Contributions for {key}', xlabel='Feature Index')
	return contrib_df


def plot_correlations(key, case_idx, m_idx, op_idx, traj_axs, traj_labels, simulator, data_processor):
	scatter_size = 0.1
	op_label = key
	for ax_idx, inp_label in enumerate(traj_labels[key]):
		inp_idx = data_processor.ctrl_inpt_cols[key].index(inp_label)
		# Plot trajectories over training phase
		traj_axs[ax_idx + m_idx, 0].scatter(data_processor.u_train[key][case_idx][:, inp_idx],
											simulator.x_sim_train[:, op_idx],
											linestyle="dashed", s=scatter_size,
											label="Model Simulation")
		traj_axs[ax_idx + m_idx, 0].scatter(data_processor.u_train[key][case_idx][:, inp_idx],
											data_processor.x_train[key][case_idx][:, op_idx],
											linestyle="solid", s=scatter_size,
											label="True Simulation")
		traj_axs[ax_idx + m_idx, 0].set(xlabel=f"{inp_label}", ylabel=f"{op_label}", title="Training Phase")
		traj_axs[ax_idx + m_idx, 0].legend()

		# Plot trajectories over testing phase
		traj_axs[ax_idx + m_idx, 1].scatter(data_processor.u_test[key][case_idx][:, inp_idx],
											simulator.x_sim_test[:, op_idx],
											linestyle="dashed", s=scatter_size,
											label="Model Simulation")
		traj_axs[ax_idx + m_idx, 1].scatter(data_processor.u_test[key][case_idx][:, inp_idx],
											data_processor.x_test[key][case_idx][:, op_idx],
											linestyle="solid", s=scatter_size,
											label="True Simulation")
		traj_axs[ax_idx + m_idx, 1].set(xlabel=f"{inp_label}", ylabel=f"{op_label}", title="Testing Phase")
		traj_axs[ax_idx + m_idx, 1].legend()


def plot_time_series(data_processor, model_name, op_idx, t_true, x_true, t_sim, x_sim, ax):

	ax.plot(t_sim, x_sim, color=COLOR_1, linestyle="dashed", label="Model Simulation")
	ax.plot(t_true, x_true, color=COLOR_2, linestyle="solid", label="True Simulation")
	# ax[-1].set(xlabel='Time [s]')
	# ax[0].set(ylabel=f'MW' if model_name == 'P_motor_partial' else 'rad')


def plot_bldpitch_drvts_ts(test_idx, data_processor,
						   BldPitch1_model_unfilt, BldPitch1_model_filt, BldPitch1_drvts_true):
	BldPitch1_drvts_ts_fig, BldPitch1_drvts_ts_axs = plt.subplots(3, 2, figsize=FIGSIZE, sharex=True)

	modeled_start_idx = 100

	time_model = BldPitch1_model_unfilt[test_idx]['time']
	time_true = BldPitch1_drvts_true['time']

	for d, drvt in enumerate(['BldPitch1', 'BldPitch1_dot', 'BldPitch1_ddot']):
		BldPitch1_drvts_ts_axs[d][0].plot(time_model[modeled_start_idx:],
										  BldPitch1_model_unfilt[test_idx][drvt][modeled_start_idx:],
										  linestyle="dashed", color=COLOR_1,
										  label="Unfiltered Modeled BldPitch1")
		BldPitch1_drvts_ts_axs[d][1].plot(time_model[modeled_start_idx:],
										  BldPitch1_model_filt[test_idx][drvt][modeled_start_idx:],
										  linestyle="dashed", color=COLOR_1,
										  label="Filtered Modeled BldPitch1")
		BldPitch1_drvts_ts_axs[d][0].plot(time_true, BldPitch1_drvts_true[drvt],
										  linestyle="solid", color=COLOR_2,
										  label="True BldPitch1")
		BldPitch1_drvts_ts_axs[d][1].plot(time_true, BldPitch1_drvts_true[drvt],
										  linestyle="solid", color=COLOR_2,
										  label="True BldPitch1")
		BldPitch1_drvts_ts_axs[d][0].set(ylabel=drvt)

	BldPitch1_drvts_ts_axs[0][0].legend()
	BldPitch1_drvts_ts_axs[0][1].legend()

	BldPitch1_drvts_ts_axs[-1][0].set(xlabel='$k$')
	return BldPitch1_drvts_ts_fig, BldPitch1_drvts_ts_axs


def plot_model_scores(key, scores, test_indices):
	fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, sharex=True)
	width = 0.25
	ind = np.arange(1, len(test_indices) + 1)
	xticks = [ascii_lowercase[i] for i in range(len(test_indices))] # ind[1::1]

	ax.bar(xticks, scores[test_indices], width, color=COLOR_1)
	ylim = (np.nanmin(scores[test_indices]) * 1, np.nanmax(scores[test_indices]) * 1)
	ax.set(xlabel='Test Dataset', xticks=xticks, ylim=ylim)
	return fig, ax


def plot_pmotor_metric_score(fig, ax, metric_scores, plotting_indices):
	# plot true and modeled max and min vs testing dataset
	width = 0.25
	ind = 1 + np.arange(len(plotting_indices))

	# mean_score_ax = ax[0]
	# max_score_ax = ax[1]
	# # max_score_ax = mean_score_ax.twinx()
	#
	# mean_score_ax.bar(ind - (width / 2),
	# 				  [metric_scores['mean'][f'seed_true'][i] for i in plotting_indices],
	# 				  width, label='True', color=COLOR_1)
	# mean_score_ax.bar(ind + (width / 2),
	# 				  [metric_scores['mean'][f'seed_partial'][i] for i in plotting_indices],
	# 				  width, label='Modeled', color=COLOR_2)
	# max_score_ax.bar(ind - (width / 2),
	# 				 [metric_scores['max'][f'seed_true'][i] for i in plotting_indices],
	# 				 width, label='True', color=COLOR_1)
	# max_score_ax.bar(ind + (width / 2),
	# 				 [metric_scores['max'][f'seed_partial'][i] for i in plotting_indices],
	# 				 width, label='Modeled', color=COLOR_2)
	#
	# mean_score_ax.set(title='True & Modeled Mean Power for each Test Dataset')
	# max_score_ax.set(title='True & Modeled Maximum Power for each Test Dataset',
	# 				 xticks=ind, xlabel='Test Dataset Index')
	# # mean_score_ax.legend()
	# max_score_ax.legend()
	
	xticks = [ascii_lowercase[i] for i in plotting_indices] # ind[1::1]
	
	rel_mean_error = [(metric_scores['mean'][f'seed_partial'][i] - metric_scores['mean'][f'seed_true'][i])
					  / metric_scores['mean'][f'seed_true'][i] for i in plotting_indices]
	rel_max_error = [(metric_scores['max'][f'seed_partial'][i] - metric_scores['max'][f'seed_true'][i])
					  / metric_scores['max'][f'seed_true'][i] for i in plotting_indices]
	ax.bar(ind - (width / 2), rel_mean_error, width, label='Mean', color=COLOR_1)
	ax.bar(ind + (width / 2), rel_max_error, width, label='Maximum', color=COLOR_2)
	ax.set(
		# title='Mean (blue) & Maximum (orange) Power Relative \n Modeling Error for each Test Dataset',
		   xticks=xticks, xlabel='Test Dataset',
		   yticklabels=np.array(ax.get_yticks()) * 100, ylabel='%')
	# ax.legend()

	fig.show()


def plot_bldpitch1_score(data_processor, test_indices, BldPitch1_model_unfilt, BldPitch1_model_filt=None):
	BldPitch1_drvts_score_fig, BldPitch1_drvts_score_axs = plt.subplots(len(data_processor.state_cols['BldPitch1']), 1,
																		figsize=FIGSIZE, sharex=True)

	if hasattr(BldPitch1_drvts_score_axs, 'shape'):
		ax = BldPitch1_drvts_score_axs
	else:
		ax = [BldPitch1_drvts_score_axs]

	BldPitch1_drvts_score = {f'{l}_unfilt': [] for l in data_processor.state_cols['BldPitch1']}

	if BldPitch1_model_filt is not None:
		BldPitch1_drvts_score = {**BldPitch1_drvts_score,
								 **{f'{l}_filt': [] for l in data_processor.state_cols['BldPitch1']}}

	for case_idx in test_indices:
		if case_idx in data_processor.training_idx:
			x_true = data_processor.x_train['BldPitch1'][np.argwhere(data_processor.training_idx == case_idx).squeeze()]
		elif case_idx in data_processor.testing_idx:
			x_true = data_processor.x_test['BldPitch1'][
				np.argwhere(data_processor.testing_idx == case_idx).squeeze()]

		for signal in ['unfilt', 'filt']:
			# if not comparing derivatives derived from filtered vs unfiltered BldPitch1 signal, skip the filtered plotts
			if BldPitch1_model_filt is None and signal == 'filt':
				continue
			for drvt in data_processor.state_cols['BldPitch1']:
				BldPitch1_drvts_score[f'{drvt}_{signal}'].append(r2_score(
					y_true=x_true[:, data_processor.state_cols['BldPitch1'].index(drvt)],
					y_pred=
					BldPitch1_model_unfilt[drvt][case_idx][1].squeeze() if signal == 'unfilt' else
					BldPitch1_model_filt[drvt][case_idx][1].squeeze()))

	width = 0.25
	ind = np.array(range(1, len(test_indices) + 1))
	for drvt_idx, drvt in enumerate(data_processor.state_cols['BldPitch1']):

		ax[drvt_idx].bar(ind, BldPitch1_drvts_score[f'{drvt}_unfilt'], width,
						 label='Unfiltered Modeled BldPitch1', color=COLOR_1)
		# if comparing derivatives derived from filtered vs unfiltered BldPitch1 signal, plot the filtered case
		if BldPitch1_model_filt is not None:
			ax[drvt_idx].bar(ind + width, BldPitch1_drvts_score[f'{drvt}_filt'], width,
							 label='Filtered Modeled BldPitch1', color=COLOR_1)

		ax[drvt_idx].set(ylabel=f'{drvt} $R^2$')

	ax[-1].set(xlabel='Case Number', xticks=np.arange(1, len(test_indices) + 1))
	# ax[-1].legend()

	return BldPitch1_drvts_score_fig, BldPitch1_drvts_score_axs, pd.DataFrame(BldPitch1_drvts_score)


# def plot_model_score(x_train, x_test, x_sim_full, state_cols, model):
def plot_model_score(fig, ax, scores, test_indices):
	n_datasets = len(x_train[model])
	n_vars = len(state_cols[model])

	width = 0.25
	ind = np.array(range(1, n_datasets + 1))

	score_fig, score_axs = plt.subplots(n_vars, 1, sharex=True)

	for i in range(n_vars):
		data_true = [np.concatenate([x_train[model][case_idx][:, i], x_test[model][case_idx][:, i]])
					 for case_idx in range(n_datasets)]

		data_modeled = [x_sim_full[case_idx][:, i] for case_idx in range(n_datasets)]

		score_axs[i].bar(ind, r2_score(y_true=data_true, y_pred=data_modeled), width, label=state_cols[i],
						 color=COLOR_1)

	score_axs[-1].set(xlabel='Dataset')
	score_axs[0].legend()
	# score_axs[0].set(title='$R^2$')


def plot_fft(y, dt, op_label, freq_1p_range):
	Y = fft(y)
	n_samples = len(y)
	T = n_samples * dt
	# freq = fftfreq(n_samples, dt)
	freq = (1 / (dt * n_samples)) * np.arange(0, (n_samples / 2) + 1)
	freq_2p_range = 2 * freq_1p_range

	two_sided_spectrum = np.abs(Y / n_samples)  # magnitudes normalized by number of samples i.e. relative frequencies
	single_sided_spectrum = two_sided_spectrum[:int(n_samples / 2) + 1]  # first half of absolute normalized frequencies
	single_sided_spectrum[1:-1] = 2 * single_sided_spectrum[1:-1]  # double middle components

	fig, ax = plt.subplots(figsize=FIGSIZE)
	ax.stem(freq, single_sided_spectrum, COLOR_1, markerfmt=" ", basefmt="-b")
	y_min = 0
	y_max = 110
	ax.set(xlabel="Frequency (Hz)", ylabel=f"FFT Amplitude of {op_label}", xlim=[0.002, 0.5], ylim=[y_min, y_max])

	rec_1p = Rectangle((freq_1p_range[0], y_min), freq_1p_range[1] - freq_1p_range[0], y_max - y_min, alpha=0.2)
	rec_2p = Rectangle((freq_2p_range[0], y_min), freq_2p_range[1] - freq_2p_range[0], y_max - y_min, alpha=0.2)

	ax.add_patch(rec_1p)
	ax.add_patch(rec_2p)

	ax.text(np.mean(freq_1p_range), y_max - 5, '1P', horizontalalignment='center', verticalalignment='bottom',
			transform=ax.transData)
	ax.text(np.mean(freq_2p_range), y_max - 5, '2P', horizontalalignment='center', verticalalignment='bottom',
			transform=ax.transData)

	plt.show()

	return fig, ax


def plot_pmotor_stats(fig, axs, P_motor_stats, data_processor):
	width = 0.25
	ind = np.array(data_processor.vmean_list)

	axs = axs.flatten()

	axs[0].bar(ind - (width / 2), P_motor_stats['mean']['vmean_true'], width, color=COLOR_1,
			   label="True")
	axs[0].bar(ind + (width / 2), P_motor_stats['mean']['vmean_partial'], width, color=COLOR_2,
			   label="Model")
	axs[0].set(ylabel='Mean Power [MW]')

	axs[1].bar(ind - (width / 2), P_motor_stats['max']['vmean_true'], width, color=COLOR_1,
			   label="True")
	axs[1].bar(ind + (width / 2), P_motor_stats['max']['vmean_partial'], width, color=COLOR_2,
			   label="Model")
	# axs[1].set(title='Max Power, $\\bar{P}_{motor}$ [MW]')
	axs[0].legend()

	# P_mean_ax.bar(ind + 2 * width, P_motor_mean['vmean_full'][::n_seed], width, color='b', label="Full Model")
	axs[-1].set(xticks=data_processor.vmean_list, xlabel="Mean Wind Speed, $\\bar{V}$")


# axs[0].set(ylabel="Mean Motor Power, $\\bar{P}_{motor}$")


def plot_features(data_processor, key, feat_ax, vmean_test, plotting_indices, model_per_wind_speed):
	num_plots = len(plotting_indices)
	if 'BldPitch1' in key:
		key = f'BldPitch1_{vmean_test}' if model_per_wind_speed else 'BldPitch1'

	# plot training data for test indices
	for test_idx, test in enumerate(plotting_indices):
		n_states = len(data_processor.state_cols[key])
		dataset_type = 'train' if test in data_processor.training_idx else 'test'
		dataset_type_idx = np.argwhere(data_processor.training_idx == test).squeeze() if dataset_type == 'train' \
			else np.argwhere(data_processor.testing_idx == test).squeeze()

		if hasattr(feat_ax, 'shape'):
			row_idx = test_idx // int(num_plots ** 0.5)
			col_idx = test_idx % int(num_plots ** 0.5)
			ax = feat_ax[row_idx, col_idx]
		else:
			ax = feat_ax

		# plot states
		for f, feat in enumerate(data_processor.feature_cols[key][:n_states]):
			if dataset_type == 'train':
				t = data_processor.t_train
				x = data_processor.x_train[key][dataset_type_idx][:, f]
			elif dataset_type == 'test':
				t = data_processor.t_test
				x = data_processor.x_test[key][dataset_type_idx][:, f]

			ax.plot(t, x, label=feat)

		# plot control inputs
		for f, feat in enumerate(data_processor.feature_cols[key][n_states:]):
			if dataset_type == 'train':
				t = data_processor.t_train
				u = data_processor.u_train[key][dataset_type_idx][:, f]
			elif dataset_type == 'test':
				t = data_processor.t_test
				u = data_processor.u_test[key][dataset_type_idx][:, f]

			ax.plot(t, u, label=feat)

		ax.set(title=f'Vmean={data_processor.vmean[dataset_type][dataset_type_idx]}, '
					 f'Seed={data_processor.seed[dataset_type][dataset_type_idx]}')

		ax.legend()
