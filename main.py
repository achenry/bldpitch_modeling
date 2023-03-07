import pysindy as ps
from collections import defaultdict
from numpy.random import choice
from scipy import stats
import time
from pysindy.utils import validate_control_variables, drop_nan_rows, drop_random_rows
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from math import comb
# Notes
# less noise for CT model
# better to include previous states in inputs
# first 2 wind speeds make for poor training data - model different for above/below rated wind speed
# TESTING: poly_fourier vs fourier lib; filtered vs unfiltered wind spedd,rotspeed

## TESTING
# CT, poly 1, Median Training Score
# DT, poly 1, Median Training Score 0.99
# CT, poly 2, Median Training Score
# DT, poly 2, Median Training Score

# ignore user warnings
import warnings

from helper_functions import *
from plotting_functions import *
from DataProcessor import DataProcessor
from Simulator import Simulator
from sklearn.metrics import r2_score, mean_squared_error
from pysindy import SINDyOptimizer
from sklearn.pipeline import Pipeline
from itertools import product

RUN_THR = True
COMPARE_THR = True

MODEL_NAMES = ['BldPitch1']  # TEST 1
# MODEL_NAMES = ['P_motor_partial']  # TEST 2
MODEL_NAMES = ['BldPitch1', 'P_motor_partial']  # TEST 2

RUN_BLDPITCH1_SIM = True
RUN_P_MOTOR_PARTIAL_SIM = True
RERUN_P_MOTOR_PARTIAL_SIM = True
RECOMPUTE_BLDPITCH_DRVTS = True
RERUN_BLDPITCH_SIMS = True
RELOAD_DATA = True
REPROCESS_DATA = True
MODEL_PER_WIND_SPEED = False
QUICK_RUN = False
USE_FILT = True
NUM_SIMULATIONS = -1
PROPORTION_TRAINING_DATA = 0.8
TEST_ON_TRAINING_DATA = False
NUM_SEEDS = 100
IS_DYNAMICAL = {'P_motor_partial': False, 'BldPitch1': True}
IS_DISCRETE = {'P_motor_partial': True, 'BldPitch1': True}  # True only works for low polyorder < 3
DEBUG = False
ENSEMBLE_FUNC = lambda coeffs: np.median(coeffs, axis=0)
PARALLEL = True
SCALER = lambda: MinMaxScaler() #MaxAbsScaler()
# SATURATION_RANGES = {'BldPitch1': (0, 0.35), 'BldPitch1_dot': (-0.03, 0.055), 'P_motor_partial': (0, None)}
SATURATION_RANGES = {'BldPitch1': (None, None), 'BldPitch1_dot': (-0.03, 0.055), 'P_motor_partial': (None, None)}

warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

V_rated = 10
GenSpeed_rated = 4.8 * 2 * np.pi / 60

# -8, -6, -3
THRESHOLD = {'P_motor_partial': np.logspace(-8, 3, 12) if not DEBUG else [1e-8],
             # np.logspace(-8, 3, 1 if DEBUG else 12),  # [0.01], #, 6
             'BldPitch1': np.logspace(-5, -1, 5) if not DEBUG else [1e-8]
             }  # [1e-4]} 6 # need low threshold for BldPitch1
NU = {'P_motor_partial': np.logspace(-8, 3, 12) if not DEBUG else [1e-8],
      # np.logspace(-8, 3, 1 if DEBUG else 12),  # [0.01], #, 6
      'BldPitch1': np.logspace(-4, 0, 5) if not DEBUG else [1]}
PLOTTING_INDICES = []  # [0] if DEBUG else [53, 54, 96]  # [86, 96, 98]

TESTING_INDICES = {'P_motor_partial': [0] if DEBUG else np.arange(100),
                   'BldPitch1': [0] if DEBUG else PLOTTING_INDICES}

FIG_DIR = f'/Users/aoifework/Documents/Research/learning_actuation/paper/ieeeconf/figs'
OUTLIST_PATH = '/Users/aoifework/Documents/Research/ipc_tuning/plant_setup_package/OutList.mat'


def generate_model(thr_idx, nu_idx, m_idx, data_processor):
	"""
	Generate SINDyC models and run simulations
	:param thr_idx:
	:param m_idx:
	:param data_processor:
	:return:
	"""
	model_name = MODEL_NAMES[m_idx]
	thr = THRESHOLD[model_name][thr_idx]
	nu = NU[model_name][nu_idx]
	
	dt = np.round(data_processor.t_train[1] - data_processor.t_train[0], 3)
	
	ensemble_coeffs = defaultdict(None)
	
	optimizer = ps.SR3(threshold=thr, nu=nu, thresholder='L2')
	
	model = ps.SINDy(
		discrete_time=IS_DISCRETE[model_name],  # if model_name=='BldPitch1' else True,
		optimizer=optimizer,
		feature_names=data_processor.feature_cols[model_name],
		feature_library=data_processor.full_lib[model_name]
		# differentiation_method=ps.SmoothedFiniteDifference()
	)
	
	# if not os.path.exists(os.path.join(results_dir, f'{model_name}_ensemble_coeffs.txt')):
	print(f'\nComputing model for {model_name}=f({[model.feature_names]}) with Threshold $={thr}$ and Nu $={nu}$\n')
	
	# Ensemble: sub-sample the time series
	# if (model_name == 'BldPitch1' and RERUN_BLDPITCH_SIMS) or (model_name == 'P_motor_partial' and RERUN_P_MOTOR_PARTIAL_SIM):
	if (IS_DYNAMICAL[model_name] and not IS_DISCRETE[model_name]):
		x_dot = data_processor.x_dot_train[model_name]
	elif not IS_DYNAMICAL[model_name]:
		x_dot = data_processor.x_train[model_name]
	elif IS_DISCRETE[model_name]:
		x_dot = None
	
	n_models = 20
	n_candidates_to_drop = 2
	x = data_processor.x_train[model_name]
	u = data_processor.u_train[model_name]
	t = dt
	multiple_trajectories = True
	model.library_ensemble = True
	model.ensemble = True
	n_subset = x[0].shape[-2]  # number of time points to use for ensemble, = time of single trajectory
	ensemble_aggregator = ENSEMBLE_FUNC
	unbias = True
	replace = True  # If ensemble true, whether or not to time sample with replacement.
	
	# model.fit(x=data_processor.x_train[model_name], t=dt, u=data_processor.u_train[model_name],
	#           x_dot=x_dot,
	#           multiple_trajectories=True,
	#           library_ensemble=True,
	#           n_candidates_to_drop=n_candidates_to_drop,
	#           ensemble=True,
	#           n_models=n_models, # for each n_models subset of the ts data, generate n_models with n_candidates_to_drop indices dropped
	#           ensemble_aggregator=ENSEMBLE_FUNC) # generates self.model_coef_ from self.coef_list
	
	trim_last_point = model.discrete_time and (x_dot is None)
	u = validate_control_variables(
		x,
		u,
		multiple_trajectories=multiple_trajectories,
		trim_last_point=trim_last_point,
	)
	model.n_control_features_ = u.shape[1]
	model.feature_library.num_trajectories = len(x)
	x, x_dot = model._process_multiple_trajectories(x, t, x_dot)
	
	# Append control variables
	x = np.concatenate((x, u), axis=1)
	
	# Drop rows where derivative isn't known unless using weak PDE form
	x, x_dot = drop_nan_rows(x, x_dot)
	
	optimizer = SINDyOptimizer(model.optimizer, unbias=unbias)
	
	model.scaler_model = Pipeline([("features", model.feature_library), ("scaler", SCALER())])
	model.scaler_model.fit(x, x_dot)
	# model.scaler_model.named_steps['scaler'].scale_ # use to transform features
	
	model.feature_library.library_ensemble = model.library_ensemble
	(model.feature_library).fit(x)
	model.coef_list = []
	for i in range(n_models):
		x_ensemble, x_dot_ensemble = drop_random_rows(
			x,
			x_dot,
			n_subset,
			replace,
			model.feature_library,
			None,
			multiple_trajectories,
		)
		for j in range(n_models):
			model.feature_library.ensemble_indices = np.sort(
				np.random.choice(
					range(model.feature_library.n_output_features_),
					n_candidates_to_drop,
					replace=False,
				)
			)
			
			inc_ensemble_indices = [i for i in range(model.feature_library.n_output_features_) if
			                        i not in model.feature_library.ensemble_indices]
			ensemble_scaler = SCALER()
			ensemble_scaler.n_features_in_ = len(inc_ensemble_indices)
			
			if ensemble_scaler.__str__() == 'StandardScaler()':
				ensemble_scaler.mean_ = model.scaler_model.named_steps['scaler'].mean_[inc_ensemble_indices]
				ensemble_scaler.scale_ = model.scaler_model.named_steps['scaler'].scale_[inc_ensemble_indices]
				ensemble_scaler.var_ = model.scaler_model.named_steps['scaler'].var_[inc_ensemble_indices]
			elif ensemble_scaler.__str__() == 'MinMaxScaler()':
				ensemble_scaler.min_ = model.scaler_model.named_steps['scaler'].min_[inc_ensemble_indices]
				ensemble_scaler.scale_ = model.scaler_model.named_steps['scaler'].scale_[inc_ensemble_indices]
				# ensemble_scaler.feature_range_ = model.scaler_model.named_steps['scaler'].feature_range_[inc_ensemble_indices]
				ensemble_scaler.data_min_ = model.scaler_model.named_steps['scaler'].data_min_[inc_ensemble_indices]
				ensemble_scaler.data_max_ = model.scaler_model.named_steps['scaler'].data_max_[inc_ensemble_indices]
				ensemble_scaler.data_range_ = model.scaler_model.named_steps['scaler'].data_range_[inc_ensemble_indices]
			elif ensemble_scaler.__str__() == 'MaxAbsScaler()':
				ensemble_scaler.scale_ = model.scaler_model.named_steps['scaler'].scale_[inc_ensemble_indices]
				ensemble_scaler.max_abs_ = model.scaler_model.named_steps['scaler'].max_abs_[inc_ensemble_indices]
			
			if True and "1" in model.feature_library.get_feature_names() and model.feature_library.get_feature_names().index(
				"1") in inc_ensemble_indices:
				# get index of constant feature in reduced feature set
				ensemble_const_idx = inc_ensemble_indices.index(model.feature_library.get_feature_names().index("1"))
				
				if ensemble_scaler.__str__() == 'StandardScaler()':
					ensemble_scaler.mean_[ensemble_const_idx] = 0
					ensemble_scaler.scale_[ensemble_const_idx] = 1
					ensemble_scaler.var_[ensemble_const_idx] = 1
				elif ensemble_scaler.__str__() == 'MinMaxScaler()':
					ensemble_scaler.min_[ensemble_const_idx] = 0
					ensemble_scaler.scale_[ensemble_const_idx] = 1
					# ensemble_scaler.feature_range_[ensemble_const_idx] = 0
					ensemble_scaler.data_min_[ensemble_const_idx] = 0
					ensemble_scaler.data_max_[ensemble_const_idx] = 0
					ensemble_scaler.data_range_[ensemble_const_idx] = 0
				elif ensemble_scaler.__str__() == 'MaxAbsScaler()':
					ensemble_scaler.scale_[ensemble_const_idx] = 1
					ensemble_scaler.max_abs_[ensemble_const_idx] = 0
			
			ensemble_scaler.n_samples_seen_ = model.scaler_model.named_steps['scaler'].n_samples_seen_
			
			model.model = Pipeline(
				[("features", model.feature_library),
				 ("scaler", ensemble_scaler),
				 ("model", optimizer)])
			
			model.model.fit(x_ensemble, x_dot_ensemble)
			coef_partial = model.model.steps[-1][1].coef_
			for k in range(n_candidates_to_drop):
				coef_partial = np.insert(
					coef_partial,
					model.feature_library.ensemble_indices[k],
					0,
					axis=-1,
				)
			model.coef_list.append(coef_partial)
	
	model.model = Pipeline(
		[("features", model.feature_library),
		 ("scaler", model.scaler_model.named_steps['scaler']),
		 ("model", optimizer)])
	model.model_coef_ = ensemble_aggregator(model.coef_list)
	
	# ensemble_coeffs[model_name] = ENSEMBLE_FUNC(model.coef_list)
	# state_feature_indices = [i for i in range(model.n_output_features_) if model.feature_names[0] in model.get_feature_names()[i]]
	# [i for i in range(n_models**2) if all([model.coef_list[i][0][j] == 0 for j in state_feature_indices])]
	model.optimizer.coef_ = model.model_coef_
	
	model.n_features_in_ = model.model.steps[0][1].n_features_in_
	model.n_output_features_ = model.model.steps[0][1].n_output_features_
	
	# model.feature_library.ensemble_indices = np.arange(model.n_output_features_)
	model.feature_library.library_ensemble = False  # necessary for predict methods later, st full library is used
	# model.n_output_features_ == model.get_feature_names().__len__()
	model.print()
	
	if model_name == 'P_motor_partial' and not IS_DYNAMICAL['P_motor_partial']:
		# just do linear regression, not a dynamical system
		coeffs = []
		n_models = 20
		n_subset = len(data_processor.t_train)
		linreg_optimizer = SINDyOptimizer(optimizer, unbias=True)
		
		XU = np.zeros((0, data_processor.n_features[model_name]))
		for dataset_idx in range(data_processor.n_training_datasets):
			XU = np.vstack(
				[XU, np.hstack([data_processor.x_train[model_name][dataset_idx],
				                data_processor.u_train[model_name][dataset_idx]])])
		
		Y = XU[:, :len(data_processor.state_cols[model_name])]
		
		model.scaler_model = Pipeline([("features", model.feature_library), ("scaler", SCALER())])
		model.scaler_model.fit(XU, Y)
		
		if "1" in model.feature_library.get_feature_names():
			# get index of constant feature in reduced feature set
			const_idx = model.get_feature_names().index("1")
			model.scaler_model.named_steps['scaler'].mean_[const_idx] = 0
			model.scaler_model.named_steps['scaler'].scale_[const_idx] = 1
		
		# XU_trans = np.zeros((0, len(model.get_feature_names())))
		for i in range(n_models):
			# Choose random n_subset points to use
			rand_inds = np.sort(choice(range(XU.shape[0]), n_subset, replace=True))
			XU_subset = XU[rand_inds, :]
			# XU_trans = models.feature_library.transform(XU_select)
			Y_subset = Y[rand_inds, :]
			
			model.model = Pipeline(
				[("features", model.feature_library),
				 ("scaler", model.scaler_model.named_steps['scaler']),
				 ("model", linreg_optimizer)])
			
			model.model.fit(XU_subset, Y_subset)
			coeffs.append(model.model.steps[-1][1].coef_.squeeze())
		
		model.model = Pipeline(
			[("features", model.feature_library),
			 ("scaler", model.scaler_model.named_steps['scaler']),
			 ("model", linreg_optimizer)])
		# coeffs.append((np.linalg.pinv(XU_trans) @ Y).squeeze())
		# data_processor.x_train[model_name][dataset_idx][:, pmotor_idx]).squeeze())
		
		ensemble_coeffs = ENSEMBLE_FUNC(coeffs)
		model.optimizer.coef_ = ensemble_coeffs
		# model.feature_library.ensemble_indices = np.arange(model.n_output_features_)
		model.feature_library.library_ensemble = False
		# model.model.steps[-1][1].coef_
		print(
			f'P_motor = {" + ".join([f"{coeff:.3f} {feat_name}" for coeff, feat_name in zip(ensemble_coeffs, model.get_feature_names())])}')
	
	return model


def compute_simulation_scores(m_idx, data_processor, simulation_cases, results_dir):
	model_name = MODEL_NAMES[m_idx]
	scores = np.nan * np.ones(data_processor.n_datasets)
	n_states = len(data_processor.state_cols[model_name])
	for t_idx in range(data_processor.n_datasets):
		if simulation_cases[t_idx] is None:
			continue
		
		_, x_sim = simulation_cases[t_idx]
		
		# for each state in this model
		for op_idx in range(n_states):
			
			if t_idx in data_processor.training_idx:
				dataset_type_idx = np.argwhere(data_processor.training_idx == t_idx).squeeze()
				x_true = data_processor.x_train[model_name][dataset_type_idx][:, op_idx]
			else:
				dataset_type_idx = np.argwhere(data_processor.testing_idx == t_idx).squeeze()
				x_true = data_processor.x_test[model_name][dataset_type_idx][:, op_idx]
			
			scores[t_idx] = r2_score(x_true, x_sim)
	
	print('Median Training Simulation Score', np.nanmedian(scores[data_processor.training_idx]))
	print('Median Testing Simulation Score', np.nanmedian(scores[data_processor.testing_idx]))
	
	# save scores
	with open(os.path.join(results_dir, f'simulation_scores.txt'), 'wb') as fh:
		pickle.dump(scores, fh)
	
	return scores


def compute_model_scores(m_idx, data_processor, model, results_dir):
	model_name = MODEL_NAMES[m_idx]
	
	dt = np.round(data_processor.t_train[1] - data_processor.t_train[0], 3)
	
	scores = np.nan * np.ones(data_processor.n_datasets)
	train_case_idx = 0
	test_case_idx = 0
	
	# COMPUTE MODEL SCORES
	for case_idx in range(data_processor.n_datasets):
		
		if case_idx in data_processor.training_idx:
			if (IS_DYNAMICAL[model_name] and not IS_DISCRETE[model_name]):
				x_dot = data_processor.x_dot_train[model_name][train_case_idx]
			elif not IS_DYNAMICAL[model_name]:
				x_dot = data_processor.x_train[model_name][train_case_idx]
			elif IS_DISCRETE[model_name]:
				x_dot = None
			
			# model.feature_library.ensemble_indices = np.arange(model.n_output_features_)
			scores[case_idx] = model.score(data_processor.x_train[model_name][train_case_idx],
			                               u=data_processor.u_train[model_name][train_case_idx],
			                               x_dot=x_dot,
			                               t=dt,
			                               metric=r2_score)
			train_case_idx += 1
		elif case_idx in data_processor.testing_idx:
			if (IS_DYNAMICAL[model_name] and not IS_DISCRETE[model_name]):
				x_dot = data_processor.x_dot_test[model_name][test_case_idx]
			elif not IS_DYNAMICAL[model_name]:
				x_dot = data_processor.x_test[model_name][test_case_idx]
			elif IS_DISCRETE[model_name]:
				x_dot = None
			
			scores[case_idx] = model.score(data_processor.x_test[model_name][test_case_idx],
			                               u=data_processor.u_test[model_name][test_case_idx],
			                               x_dot=x_dot,
			                               t=dt,
			                               metric=r2_score)
			test_case_idx += 1
	
	print('Median Training Score', np.nanmedian(scores[data_processor.training_idx]))
	print('Median Testing Score', np.nanmedian(scores[data_processor.testing_idx]))
	
	# save scores
	with open(os.path.join(results_dir, f'model_scores.txt'), 'wb') as fh:
		pickle.dump(scores, fh)
	
	return scores


def simulate_model(model, scores, thr_idx, nu_idx, m_idx, data_processor,
                   num_simulations=NUM_SIMULATIONS):
	model_name = MODEL_NAMES[m_idx]
	thr = THRESHOLD[model_name][thr_idx]
	nu = NU[model_name][nu_idx]
	results_dir = f'./results/{model_name}/thr-{thr}_nu-{nu}'
	test_indices_full = [data_processor.testing_idx[idx] for idx in
	                     np.flip(np.argsort(scores[data_processor.testing_idx]))] # Must apply flip if using r2_score
	
	if num_simulations == -1:
		num_simulations = len(test_indices_full)
	
	test_indices = sorted(test_indices_full[:num_simulations])  # TESTING_INDICES[model_name]
	
	if (not RUN_BLDPITCH1_SIM or not RUN_P_MOTOR_PARTIAL_SIM) and os.path.exists(
		os.path.join(results_dir, f'simulation_cases.txt')):
		with open(os.path.join(results_dir, f'simulation_cases.txt'), 'rb') as fh:
			simulation_cases = pickle.load(fh)
	else:
		simulation_cases = {}
	
	## Simulate BldPitch1 and derivative models
	if RUN_BLDPITCH1_SIM and 'BldPitch1' in model_name:
		simulator = Simulator(model_name='BldPitch1', model=model, data_processor=data_processor)
		if RERUN_BLDPITCH_SIMS:
			simulation_cases = []
			# test_indices_tmp = test_indices_full[:NUM_SIMULATIONS]  # list(test_indices)
			test_indices_tmp = list(test_indices)
			
			next_test_idx = num_simulations
			
			tic = time.perf_counter()
			while True:
				sim_success = True
				
				simulation_cases = simulator.simulate_cases(
					test_indices_tmp,
					full_model=False,
					BldPitch1_model=None,
					computed_derivatives=False,
					is_discrete=IS_DISCRETE['BldPitch1'],
					is_dynamical=IS_DYNAMICAL['BldPitch1'],
					parallel=PARALLEL)
				# test_indices_recent = list(test_indices_tmp)
				# test_indices_final.update(test_indices_recent)
				# test_indices_tmp = []
				test_idx = 0
				for sim_full_idx, sim in enumerate(simulation_cases):
					if sim is not None:
						t_sim, x_sim = sim
						if t_sim.shape[0] != x_sim.shape[0]:
							del test_indices_tmp[test_idx]
							simulation_cases[sim_full_idx] = None
							test_indices_tmp = test_indices_tmp + test_indices_full[next_test_idx:next_test_idx + 1]
							# test_indices_final.remove(test_indices_recent[sim_idx])
							next_test_idx += 1
							sim_success = False
						test_idx += 1
				
				if sim_success:
					break
			toc = time.perf_counter()
			print(f'Ran {len(test_indices_tmp)} simulations in {toc - tic:0.4f}')
			simulator.save_simulations(results_dir, simulation_cases)
		else:
			simulation_cases = simulator.load_simulations(results_dir)
		
	if RUN_P_MOTOR_PARTIAL_SIM and 'P_motor_partial' in model_name:
		simulator = Simulator(model_name='P_motor_partial', model=model,
		                      data_processor=data_processor)
		if RERUN_P_MOTOR_PARTIAL_SIM:
			
			# if 'P_motor_partial' not in simulation_cases:
			
			simulation_cases['P_motor_partial'] = \
				simulator.simulate_cases(
					test_indices,
					full_model=False,
					BldPitch1_model=None,
					computed_derivatives=False,
					is_discrete=IS_DISCRETE['P_motor_partial'],
					is_dynamical=IS_DYNAMICAL['P_motor_partial'],
					parallel=PARALLEL)
			# simulation_cases['P_motor_partial'] = list(simulator_gens['P_motor_partial'])
			
			simulator.save_simulations(results_dir, simulation_cases['P_motor_partial'])
		else:
			simulation_cases = simulator.load_simulations(results_dir)
		
		# compute modelled P_motor mean for each seed and vmean for partial and full P_motor model
		# for each mean wind speed
		# def compute_simulation_metrics(data_processor):
		
		# for dataset_type in ['train', 'test']:
		P_motor_stats = {'seed': data_processor.seed_list,
		                 'vmean': data_processor.vmean_list,
		                 'mean': {
			                 'seed_true': data_processor.P_motor_stats['mean_seed'],
			                 'vmean_true': data_processor.P_motor_stats['mean_vmean'],
			                 'seed_partial': [],
			                 'vmean_partial': [],
			                 'seed_full': [],
			                 'vmean_full': [],
		                 },
		                 'max': {'seed_true': data_processor.P_motor_stats['max_seed'],
		                         'vmean_true': data_processor.P_motor_stats['max_vmean'],
		                         'seed_partial': [],
		                         'vmean_partial': [],
		                         'seed_full': [],
		                         'vmean_full': [],
		                         }
		                 }
		
		P_motor_score = defaultdict(list)
		for v_idx in range(data_processor.n_wind):
			# for each turbulence seed
			for s_idx in range(data_processor.n_seed):
				case_idx = (v_idx * data_processor.n_seed) + s_idx
				
				if case_idx not in test_indices:
					continue
				
				print(f'\nComputing P_motor Mean and Max for Case {case_idx}...')
				
				states_idx = 1
				
				_, x_sim_partial = simulation_cases[case_idx]
				x_modeled = simulation_cases[case_idx][states_idx][:,
				            data_processor.state_cols['P_motor_partial'].index('P_motor')]
				
				P_motor_stats['mean']['seed_partial'].append(np.mean(x_modeled))
				P_motor_stats['max']['seed_partial'].append(np.max(x_modeled))
				
				# Compute MSE between true and modeled values for each case
				P_motor_score['partial'].append(scores[case_idx])
			
			# compute P_motor_mean and max over all seeds for this vmean
			for metric in ['mean', 'max']:
				if metric == 'mean':
					# mean over all seeds at this wind speed
					new_vals = [np.mean([P_val for vmean, P_val in
										 zip(P_motor_stats['vmean'],
											 P_motor_stats[metric][f'seed_partial']) if
										 vmean == data_processor.vmean_list[v_idx]])]
				elif metric == 'max':
					# maximum over all seeds at this wind speed
					new_vals = [np.max([P_val for vmean, P_val in
										zip(P_motor_stats['vmean'],
											P_motor_stats[metric][f'seed_partial'])
										if
										vmean == data_processor.vmean_list[v_idx]])]
				P_motor_stats[metric][f'vmean_partial'] \
					= P_motor_stats[metric][f'vmean_partial'] \
					  + new_vals
		
		# Plot P_motor_partial and P_motor_full R2 Score vs all test_cases
		score_fig, score_ax = plt.subplots(1, 1, figsize=FIGSIZE, sharex=True)
		plot_model_scores(score_fig, score_ax, 'Power', scores, data_processor.testing_idx)
		score_fig.savefig(os.path.join(FIG_DIR, f'inst_score_vs_dataset_thr-{thr}.png'))
		
		# plot mean and maximum motor power score vs. test dataset number for each threshold
		# P_motor_metrics_fig, P_motor_metrics_ax = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
		P_motor_metrics_fig, P_motor_metrics_ax = plt.subplots(1, 1, figsize=(FIGSIZE[0], FIGSIZE[1] / 2))
		# TODO replace this plot with R2
		plot_pmotor_metric_score(P_motor_metrics_fig, P_motor_metrics_ax, P_motor_stats, data_processor.testing_idx)
		P_motor_metrics_fig.savefig(os.path.join(FIG_DIR, f'metric_score_vs_dataset_thr-{thr}.png'))
	
	# for each state in this model, plot the contributions of each term and coefficient ensemble values
	n_states = len(data_processor.state_cols[model_name])
	if False:
		for op_idx in range(n_states):
			ensemble_fig, ensemble_axs = plt.subplots(n_states, 1, figsize=FIGSIZE)
			ensemble_coeffs_df = plot_ensemble_error(model, data_processor.state_cols[model_name], op_idx, ensemble_axs)
			
			ensemble_coeffs_df.to_csv(os.path.join(results_dir, f'{model_name}_{op_idx}_ensemble_coefficients.csv'))
			
			# plot the relative contributions of different terms, averaged over all datasets
			contribution_fig, contribution_axs = plt.subplots(n_states, 1, figsize=FIGSIZE)
			contribs_df = plot_contribution_error(model, model_name, op_idx, contribution_axs, data_processor,
			                                      TEST_ON_TRAINING_DATA, parallel=PARALLEL)
			# parallel=PARALLEL if model_name=='P_motor_partial' else False)
			contribs_df.to_csv(os.path.join(results_dir, f'{model_name}_{op_idx}_contributions.csv'))
			contribution_fig.show()
			
			# Plot barchart of coefficients for different terms for BlPitch1 and P_motor
			coeff_fig, coeff_axs = plt.subplots(n_states, 1, figsize=FIGSIZE)
			coeffs_df = plot_coeffs(model, model_name, op_idx, coeff_axs)
			coeffs_df.to_csv(os.path.join(results_dir, f'{model_name}_{op_idx}_coefficients.csv'))
		
		ensemble_fig.show()
		ensemble_fig.savefig(os.path.join(fig_dir, f'ensemble_fig_thr-{thr}.png'))
		
		coeff_fig.show()
		coeff_fig.savefig(os.path.join(fig_dir, f'coeff_fig_thr-{thr}.png'))
	
	for t_idx, t in enumerate(PLOTTING_INDICES):
		
		if model_name == 'P_motor_partial' and RUN_P_MOTOR_PARTIAL_SIM:
			# Plot barchart of true vs partial vs full P_mean for each vmean
			if t_idx == 0:
				P_mean_fig, P_mean_axs = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
			plot_pmotor_stats(P_mean_fig, P_mean_axs, P_motor_stats, data_processor)
		
		t_sim, x_sim = simulation_cases[model_name][t]
		
		# for each state in this model
		for op_idx in range(n_states):
			# Plot Simulated, True Pmotor, Beta vs time
			if t_idx == 0:
				sim_ts_fig, sim_ts_axs = plt.subplots(NUM_SIMULATIONS, 1,
				                                      sharex=True, figsize=FIGSIZE)
			
			# op_label = data_processor.state_cols[model_name][op_idx]
			
			if t_idx in data_processor.training_idx:
				dataset_type_idx = np.argwhere(data_processor.training_idx == t_idx).squeeze()
				x_true = data_processor.x_train[model_name][dataset_type_idx][:, op_idx]
				t_true = data_processor.t_train
			else:
				dataset_type_idx = np.argwhere(data_processor.testing_idx == t_idx).squeeze()
				x_true = data_processor.x_test[model_name][dataset_type_idx][:, op_idx]
				t_true = data_processor.t_test
			plot_time_series(model_name, t_idx, op_idx, t_true, x_true, t_sim, x_sim,
			                 sim_ts_axs[t_idx] if hasattr(sim_ts_axs, 'shape') else sim_ts_axs)
			
			if t_idx == len(PLOTTING_INDICES) - 1:
				sim_ts_fig.show()
				sim_ts_fig.savefig(os.path.join(fig_dir, f'{model_name}_sim_ts_fig_thr-{thr}.png'))
	
	if model_name == 'P_motor_partial' and RUN_P_MOTOR_PARTIAL_SIM:
		P_mean_fig.show()
		P_mean_fig.savefig(os.path.join(fig_dir, f'P_mean_fig_thr-{thr}.png'))
	
	# Plot Pmotor vs Beta, Beta vs V,
	# plot_correlations(model_name, test_idx, m_idx, op_idx, traj_axs, traj_labels, ctrl_inpt_cols, x_sim_train, x_sim_test,
	#                   x_train, x_test, u_train, u_test)
	
	# for each metric, each case (every seed of every wind speed) and each vmean, each model type, \
	# compute score to compare across threshold values
	if RUN_P_MOTOR_PARTIAL_SIM and 'P_motor_partial' in model_name:
		P_motor_score = {}
		
		for metric in ['mean', 'max']:
			P_motor_score[metric] = {}
			for group in ['seed']:
				for model_type in ['partial']:
					P_motor_score[metric][f'{group}_{model_type}'] = np.zeros(data_processor.n_datasets)
					for dataset_type in ['train', 'test']:
						idx = data_processor.testing_idx if dataset_type == 'test' else data_processor.training_idx
						P_motor_score[metric][f'{group}_{model_type}'][idx] \
							= r2_score(y_true=np.array(P_motor_stats[metric][f'{group}_true'])[idx],
							           y_pred=np.array(P_motor_stats[metric][f'{group}_{model_type}'])[idx])
		# = compute_mse(P_motor_stats[metric][f'{group}_true_{dataset_type}'],
		# 			  P_motor_stats[metric][f'{group}_{model_type}'])
		
		# pd.DataFrame(P_motor_stats).to_csv(os.path.join(results_dir, f'P_motor_stats.csv'))
		with open(os.path.join(results_dir, 'P_motor_score.txt'), 'wb') as fh:
			pickle.dump(P_motor_score, fh)
	
	return simulation_cases


VMEAN_LIST = [14]
SEED_LIST = list(range(0, NUM_SEEDS, 1))  # seeds in data


def main():
	## Load Training Data
	
	# data_dir = '/Users/aoifework/Documents/Research/learning_actuation/simulations'
	# data_suffix = 'out'
	data_dir = '/Users/aoifework/Documents/Research/learning_actuation/simulation_output'
	data_suffix = 'mat'
	data_prefix = 'AD_SOAR_c7_V2f_c73_Clean_newIPC_'
	
	# MODEL_NAMES = ['BldPitch1', 'BldPitch1_dot', 'BldPitch1_ddot', 'P_motor_partial']
	# MODEL_NAMES = ['BldPitch1']
	# 1+ ([f'BldPitch1_{v}' for v in vmean_list] if MODEL_PER_WIND_SPEED else ['BldPitch1'])
	
	data_processor = DataProcessor(vmean_list=VMEAN_LIST, seed_list=SEED_LIST,
	                               V_rated=V_rated,
	                               model_names=MODEL_NAMES,
	                               x_saturation=SATURATION_RANGES,
	                               model_per_wind_speed=MODEL_PER_WIND_SPEED)
	
	if RELOAD_DATA:
		data_processor.load_all_datasets(data_dir, outlist_path=OUTLIST_PATH,
		                                 data_prefix=data_prefix, data_suffix=data_suffix,
		                                 parallel=PARALLEL, use_filt=USE_FILT)
		data_processor.save_data_all()
	else:
		data_processor.load_data_all()
	
	if REPROCESS_DATA or RELOAD_DATA:
		PLOTTING_INDICES = []
		data_processor.split_all_datasets(proportion_training_data=PROPORTION_TRAINING_DATA,
		                                  testing_idx=PLOTTING_INDICES)
		data_processor.process_data()
		data_processor.save_processed_data()
	# del data_processor.data_all, data_processor.data_all_standardized
	else:
		data_processor.load_processed_data()
	
	# plot features
	# vmean_test = 14
	# NUM_SIMULATIONS = 4
	# PLOTTING_INDICES = [8, 9, 10, 11]
	# feat_fig, feat_ax = plt.subplots(int(NUM_SIMULATIONS ** 0.5), int(NUM_SIMULATIONS ** 0.5), sharex=True)
	# plot_features(data_processor, 'BldPitch1', feat_ax, vmean_test, PLOTTING_INDICES, MODEL_PER_WIND_SPEED)
	# feat_fig.show()
	# feat_fig.savefig(os.path.join(FIG_DIR, f'{model_name}_feat_fig.png'))
	
	if RUN_THR:
		# dt = data_processor.t_train[1] - data_processor.t_train[0]
		# freq_1p_range = np.array(
		# 	[data_processor.data_all[0]['GenSpeed'].abs().min(), data_processor.data_all[0]['GenSpeed'].abs().max()]) * (
		# 					1 / (2 * np.pi))
		# fft_fig, fft_ax = plot_fft(data_processor.data_all[0].loc[:, 'RootMzc1'].abs().to_numpy(), dt, '$M_{z, c1}$',
		# 						   freq_1p_range)
		# fft_fig.savefig(os.path.join('./figs', f'fft_fig.png'))
		
		# Choose case index for plotting
		# vmean_test = 12
		# seed_test = 2
		# test_indices = [seed_idx for seed_idx in
		#                 [idx for idx, vmean in enumerate(data_processor.vmean_train + data_processor.vmean_test) if vmean == vmean_test]
		#                 if
		#                 seed_idx in [idx for idx, seed in enumerate(data_processor.seed_train + data_processor.seed_test) if seed == seed_test]]
		
		## Define Candidate Functions
		data_processor.generate_candidate_funcs()
	
	opt_thresholds = {'P_motor_partial': 10 ** (-1), 'BldPitch1': (0.1, 1)}
	
	## Fit the model for cont-time system
	# pool = Pool(mp.cpu_count())
	model_cases = defaultdict(list)
	sim_scores_df = {}
	for m_idx, model_name in enumerate(MODEL_NAMES):
		param_cases = list(product(range(len(THRESHOLD[model_name])), range(len(NU[model_name]))))
		for thr_idx, nu_idx in param_cases:
			results_dir = f'./results/{model_name}/thr-{THRESHOLD[model_name][thr_idx]}_nu-{NU[model_name][nu_idx]}'
			for dir in [results_dir, FIG_DIR]:
				if not os.path.exists(dir):
					os.makedirs(dir)
			# if THRESHOLD[model_name][thr_idx] == opt_thresholds[model_name][0] and NU[model_name][nu_idx] == opt_thresholds[model_name][1]:
			model = generate_model(thr_idx, nu_idx, m_idx, data_processor)
			model_scores = compute_model_scores(m_idx, data_processor, model, results_dir)
			
			# simulate test datasets
			simulations = simulate_model(model, model_scores, thr_idx, nu_idx, m_idx, data_processor, NUM_SIMULATIONS)
			simulation_scores = compute_simulation_scores(m_idx, data_processor, simulations, results_dir)
			
			model_cases[model_name].append((thr_idx, nu_idx, model, model_scores, simulations, simulation_scores))
			
			print(
				f'Figure: Blade-Pitch Model Scores for each Test Dataset with Threshold $={THRESHOLD[model_name][thr_idx]}$, Nu = $={NU[model_name][nu_idx]}$')
			score_fig, score_ax = plot_model_scores(model_name, simulation_scores, data_processor.testing_idx)
			score_fig.show()
			score_fig.savefig(os.path.join(FIG_DIR,
			                               f'{model_name}_inst_score_vs_dataset_thr-{THRESHOLD[model_name][thr_idx]}_nu-{NU[model_name][nu_idx]}.png'))
		
		# tabulate median model and simulation score over all testing indices for each model
		model_scores = [case[3][data_processor.testing_idx] for idx, case in enumerate(model_cases[model_name])]
		simulation_scores = [case[5][data_processor.testing_idx] for idx, case in
		                     enumerate(model_cases[model_name])]
		sim_scores_df[model_name] = pd.DataFrame(data={
			'Threshold': [THRESHOLD[model_name][case[1]] for case in model_cases[model_name]],
			'Nu': [NU[model_name][case[1]] for case in model_cases[model_name]],
			'MedianModelScore': [np.nanmedian(sc) for sc in model_scores],
			'MedianSimulationScore': [np.nanmedian(sc) for sc in simulation_scores],
			'TopModelScoringDatasets': [data_processor.testing_idx[[np.argsort(sc)][0]] for sc in model_scores],
			'TopSimulationScoringDatasets': [data_processor.testing_idx[[np.argsort(sc)][0]] for sc in simulation_scores]
		}).sort_values(by='MedianSimulationScore', ascending=False)  # ordered with higest R2 scores on top
		print('Scores for Parameter Sweep')
		print(sim_scores_df[model_name])
	
	# model_cases[model_name] = pool.starmap(generate_model, [(thr_idx, nu_idx, m_idx, data_processor)
	#                                                   for thr_idx, nu_idx in product(range(len(THRESHOLD[model_name])), range(len(NU[model_name])))])
	
	# pool.close()
	
	# PLOTTING_INDICES = [77, 83, 38, 78]
	
	# else:
	thresholds = np.array(THRESHOLD['BldPitch1'])
	nus = np.array(NU['BldPitch1'])
	
	# plot time-series of simulated vs true BldPitch1/Pmotor values
	# t_idx = PLOTTING_INDICES[-1] # choose one with highest score
	#
	
	output_idx = 0
	ts_fig, ts_ax = plt.subplots(2, 1, sharex=True)
	for m_idx, model_name in enumerate(MODEL_NAMES):
		opt_model_row = sim_scores_df[model_name].iloc[0]
		opt_thr = opt_model_row['Threshold']
		opt_nu = opt_model_row['Nu']
		opt_model_idx = [i for i in range(len(model_cases[model_name]))
		                 if (THRESHOLD[model_name][model_cases[model_name][i][0]] == opt_thr
		                     and NU[model_name][model_cases[model_name][i][1]] == opt_nu)][0]
		
		opt_model_simulations = model_cases[model_name][opt_model_idx][4]
		t_idx = opt_model_row['TopSimulationScoringDatasets'][-1]
		
		# get simulations from first not None simulation in model_cases
		t_sim, x_sim = opt_model_simulations[t_idx]
		if t_idx in data_processor.training_idx:
			dataset_type_idx = np.argwhere(data_processor.training_idx == t_idx).squeeze()
			x_true = data_processor.x_train[model_name][dataset_type_idx][:, output_idx]
			t_true = data_processor.t_train
		else:
			dataset_type_idx = np.argwhere(data_processor.testing_idx == t_idx).squeeze()
			x_true = data_processor.x_test[model_name][dataset_type_idx][:, output_idx]
			t_true = data_processor.t_test
		ts_ax[m_idx].set(ylabel=f'MW' if model_name == 'P_motor_partial' else 'rad')
		
		# [i for i in range(100) if
		#  opt_model_simulations[i] is not None and np.isclose(opt_model_simulations[i][1][0], x_true[0])]
			
		plot_time_series(data_processor, model_name, output_idx, t_true, x_true, t_sim, x_sim, ts_ax[m_idx])
	
	ts_ax[-1].set(xlabel='Time [s]')
	
	ts_fig.show()
	ts_fig.savefig(os.path.join(FIG_DIR, f'sim_ts_fig_thr-opt_nu-opt.png'))
	
	# get scores
	scores = defaultdict(dict)
	# {'BldPitch1': {}, 'P_motor_partial': None}
	for model_name in MODEL_NAMES:
		for thr, nu in product(thresholds, nus):
			results_dir = f'./results/{model_name}/thr-{thr}_nu-{nu}'
			with open(os.path.join(results_dir, f'model_scores.txt'), 'rb') as fh:
				scores[model_name][thr] = pickle.load(fh)
			# remove outliers
			z = np.abs(stats.zscore(scores[model_name][thr]))
			# z = stats.zscore(scores[thr])
			scores[model_name][thr][z > 3] = np.nan
	
	# plot mean and error scores vs. threshold for both models TODO plot boxplot of scores for different test datasets
	score_vs_thr_fig, left_score_vs_thr_ax = plt.subplots(1, 1, figsize=(FIGSIZE[0], FIGSIZE[1] / 2))
	right_score_vs_thr_ax = left_score_vs_thr_ax.twinx()
	ind = np.log10(thresholds).astype(int)
	width = 0.25
	# score_ax.errorbar(ind, [np.nanmean(scores[thr]['test']) for thr in thresholds],
	# 				  yerr=[np.nanstd(scores[thr]['test']) for thr in thresholds],
	# 				  fmt='o')i
	# score_ax.bar(ind - (width / 2),
	# 			 [np.nanmedian(scores[thr][data_processor.training_idx]) for thr in thresholds],
	# 			 width, label='Training Datasets', color='#1f77b4') # '#ff7f0e'
	# left_score_vs_thr_ax.bar(ind - (width / 2),
	# 			 [np.nanmedian(scores['BldPitch1'][thr][data_processor.testing_idx]) for thr in thresholds],
	# 			 width, label=r'$\beta$', color=COLOR_1)
	# right_score_vs_thr_ax.bar(ind + (width / 2),
	# 			 [np.nanmedian(scores['P_motor_partial'][thr][data_processor.testing_idx]) for thr in thresholds],
	# 			 width, label=r'$P_\beta$', color=COLOR_2)
	# TODO add colors
	left_score_vs_thr_ax.boxplot([scores['BldPitch1'][thr][data_processor.testing_idx]
	                              for thr in thresholds])
	right_score_vs_thr_ax.bar([scores['P_motor_partial'][thr][data_processor.testing_idx]
	                           for thr in thresholds], label=r'$P_\beta$')
	
	for model_name in MODEL_NAMES:
		print(f'Threshold with highest Median $R^2$ over all Training Datasets for {model_name}',
		      thresholds[
			      np.flip(
				      np.argsort(
					      [np.nanmedian(scores[model_name][thr][data_processor.training_idx]) for thr in thresholds]))
		      ])
		print(f'Threshold with highest Median $R^2$ over all Test Datasets for {model_name}',
		      thresholds[
			      np.flip(
				      np.argsort(
					      [np.nanmedian(scores[model_name][thr][data_processor.testing_idx]) for thr in thresholds]))
		      ])
	left_score_vs_thr_ax.set(
		# title=r'$\beta$ (blue, left axis) and $P_\beta$ (orange, right axis) $R^2$ Median over Test Datasets vs. Threshold',
		xlabel='Log of Threshold, $\\log_{10}\\lambda$ [-]',
		xticks=ind, xticklabels=ind,
		ylim=(0.274, 0.287))
	right_score_vs_thr_ax.set(ylim=(0.739, 0.744))
	
	# left_score_vs_thr_ax.legend()
	
	score_vs_thr_fig.savefig(os.path.join(FIG_DIR, f'model_scores_vs_thr.png'))
	score_vs_thr_fig.show()
	
	best_score_fig, left_best_score_ax = plt.subplots(1, 1, figsize=(FIGSIZE[0], 0.5 * FIGSIZE[1]))
	right_best_score_ax = left_best_score_ax.twinx()
	dataset_ind = np.arange(1, len(data_processor.testing_idx) + 1)
	
	left_best_score_ax.bar(dataset_ind - (width / 2), scores['BldPitch1'][10 ** (1)][data_processor.testing_idx], width,
	                       color=COLOR_1, label=r'$\beta$')
	right_best_score_ax.bar(dataset_ind + (width / 2),
	                        scores['P_motor_partial'][10 ** (-1)][data_processor.testing_idx], width, color=COLOR_2,
	                        label=r'$P_\beta$')
	left_best_score_ax.set(xlabel='Test Dataset Index', xticks=dataset_ind[1::2])
	# title=r'$\beta$ (blue, left axis) and $P_\beta$ (orange, right axis) $R^2$ for each Test Dataset')
	# best_score_ax.legend()
	best_score_fig.savefig(os.path.join(FIG_DIR, 'best_model_scores.png'))
	best_score_fig.show()
	
	# results_dir = f'./results/P_motor_partial/thr-{1.0}'
	# with open(os.path.join(results_dir, f'simulation_cases.txt'), 'rb') as fh:
	# 	simulation_cases = pickle.load(fh)
	#
	# best_pmotor_metric_fig, left_best_pmotor_metric_ax = plt.subplots(1, 1, figsize=(FIGSIZE[0], FIGSIZE[1] / 2))
	# right_best_pmotor_metric_ax = left_best_pmotor_metric_ax.twinx()
	
	# plot_pmotor_metric_score(fig, ax, metric_scores, plotting_indices)
	
	if model_name == 'P_motor_partial':
		
		metric_scores = {}
		for thr in thresholds:
			results_dir = f'./results/{model_name}/thr-{thr}'
			with open(os.path.join(results_dir, 'P_motor_score.txt'), 'rb') as fh:
				metric_scores[thr] = pickle.load(fh)
				# remove outliers
				for m, metric in enumerate(['mean', 'max']):
					z = np.abs(stats.zscore(metric_scores[thr][metric]['seed_partial']))
					metric_scores[thr][metric]['seed_partial'][z > 3] = np.nan
		
		# plot median of mean and maximum motor power score vs. threshold
		metric_score_fig, metric_score_ax = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
		#
		width = 0.25
		ind = np.log10(thresholds).astype(int)
		# plot mean for left yaxis and max for right yaxis
		for m, metric in enumerate(['mean', 'max']):
			# metric_score_ax[m].bar(ind - (width / 2),
			# 					   [np.nanmean(
			# 						   metric_scores[thr][metric][f'seed_partial'][data_processor.training_idx])
			# 						   for thr in thresholds], width,
			# 					   # yerr=[np.std(metric_scores[thr][metric][f'seed_partial'][data_processor.training_idx])],
			# 					   # fmt='o',
			# 					   label=f'Training Datasets')
			# plot median absolute error instead?
			metric_score_ax[m].bar(ind,
			                       [np.nanmedian(
				                       metric_scores[thr][metric][f'seed_partial'][data_processor.testing_idx])
				                       for thr in thresholds], width, color=COLOR_1,
			                       # yerr=[np.std(metric_scores[thr][metric][f'seed_partial'][data_processor.testing_idx])],
			                       # fmt='o',
			                       label=f'Test Datasets')
		
		# metric_score_ax[m].set(
		# 	title=f'Median of {metric.capitalize()} Power $R^2$ over Datasets')
		
		metric_score_ax[-1].set(xticklabels=ind,
		                        xlabel='Log of Threshold, $\\log_{10}\\lambda$ [-]',
		                        xticks=ind)
		metric_score_fig.savefig(os.path.join(FIG_DIR, f'{model_name}_mean_max_score_vs_thr'))
		plt.show()
	
	# generate csv of coefficients (mean and standard deviation) for different values of lambda
	# all_coeffs
	
	for thr_idx, nu_idx in product(thresholds, nus):
		thr = thresholds[thr_idx]
		nu = nus[nu_idx]
		results_dir = f'./results/{model_name}/thr-{thr}_nu-{nu}'
		op_idx = 0
		ensemble_coeffs_df = pd.read_csv(os.path.join(results_dir, f'{op_idx}_ensemble_coefficients.csv'))
		
		if thr_idx == 0:
			all_coeffs_df = pd.DataFrame(data={}, index=ensemble_coeffs_df['Candidate Function'])
		
		ensemble_coeffs_df = ensemble_coeffs_df.set_index('Candidate Function')
		ensemble_coeffs_df.rename(
			columns={'Mean': f'mean_{thr}',
			         'Median': f'median_{thr}',
			         'Normalized Variance': f'norm_var_{thr}'}, inplace=True)
		ensemble_coeffs_df = ensemble_coeffs_df[[f'mean_{thr}', f'median_{thr}', f'norm_var_{thr}']]
		all_coeffs_df = pd.merge(all_coeffs_df, ensemble_coeffs_df, left_index=True, right_index=True,
		                         how="outer")
	
	# TODO order from greatest to least absolute mean value for smallest value of lambda,
	#  remove all mean zero rows
	#  add \ before all underscores
	all_coeffs_df.to_csv(os.path.join('.', f'{model_name}_all_coefficients.csv'))


if __name__ == '__main__':
	main()
