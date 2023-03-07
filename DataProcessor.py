import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, lfilter, filtfilt, savgol_filter
from pysindy.feature_library import CustomLibrary, PolynomialLibrary, GeneralizedLibrary, FourierLibrary
import os
from pysindy import FiniteDifference, SmoothedFiniteDifference
import pickle
from helper_functions import generate_secord_lpf_var, generate_firstord_lpf_var, remove_spikes
from pyFAST.input_output.fast_output_file import FASTOutputFile
import multiprocessing as mp
from multiprocessing import Pool
from collections import defaultdict
from plotting_functions import plot_fft
import matplotlib.pyplot as plt
import itertools
import heapq
import mat73
from sklearn.preprocessing import StandardScaler

GBRATIO = 1
# GENSPEED_SCALING = 100
# GENSPEED_DEV_RATED_SCALING = 100

SUBSET_IN_CHANNELS = [
            'Time',
            'Wind1VelX',
            # 'Azimuth',
            'BldPitch1', 'BldPitch1_dot',
            'P_motor',
            'Wind1VelX_dev_mean',
            'Wind1VelX_dev_rated',
            'RootMzc1'
]

SUBSET_MODEL_CHANNELS = [
            'Time',
            'BldPitch1', 'BldPitch1_dot', 'BldPitch1_ddot',
            'P_motor',
            # 'Wind1VelX',
            'Wind1VelX_dev_mean',
            'Wind1VelX_dev_rated', 'Wind1VelX_dev_mean_dot'
]

# SCALING = {
#     'Wind1VelX_dev': 1,
#     'Wind1VelX_dev_rated': 1,
#     'GenSpeed_dev_rated': 10,
#     'GenSpeed': 10,
#     'BldPitch1': 1,
#     'BldPitch1_dot': 1,
#     'BldPitch1_ddot': 1,
#     'Azimuth': 1
# }

class DataProcessor:
    def __init__(self, **kwargs):
        self.vmean_list = kwargs['vmean_list']
        self.seed_list = kwargs['seed_list']
        self.V_rated = kwargs['V_rated']
        self.x_saturation = kwargs['x_saturation']

        self.data_all = {'P_motor_mean_seed': [], 'P_motor_mean_vmean': [],
                         'P_motor_max_seed': [], 'P_motor_max_vmean': [],
                         'seed': [], 'vmean': [], 'filename': [], 'dt': [],
                         'training_input_data': [], 'testing_input_data': [],
                         'training_output_data': [], 'testing_output_data': [],
                         'n_datapoints': []}


        self.n_wind = len(self.vmean_list)
        self.n_seed = len(self.seed_list)
        self.n_datasets = self.n_wind * self.n_seed
        self.n_training_datasets = None
        self.n_testing_datasets = None
        self.model_names = kwargs['model_names']
        self.model_per_wind_speed = kwargs['model_per_wind_speed']
        self.training_idx = None
        self.testing = None


    def load_dataset(self, data_filename, v_idx, s_idx, data_suffix, use_filt=True,
                     outlist_path='/Users/aoifework/Documents/Research/ipc_tuning/plant_setup_package/OutList.mat'):

        print(f'> loading {data_filename} ..\n')

        # load the data
        if data_suffix == 'mat':
            data = mat73.loadmat(data_filename)
            outlist = loadmat(outlist_path)
            outlist = [el[0][0] for el in outlist['OutList']]
            # extract the output channel names
            # op_channels = [ch[0] for ch in data['Chan'][0].dtype.descr]
            data = pd.DataFrame({outlist[i]: data['sim_case']['signals']['values'][:, i] for i in range(len(outlist)) if outlist[i] in SUBSET_IN_CHANNELS})

        elif data_suffix == 'out':
            fst_output = FASTOutputFile(data_filename)
            data = fst_output.toDataFrame()

            # create a dictionary of the data
            data.rename(columns=lambda col: col.split('_')[0], inplace=True)
            data = data[list(set(SUBSET_IN_CHANNELS) & set(data.columns))]

        # generate beta_dot and beta_ddot values
        init_idx = 5001

        data.loc[:, 'BldPitch1'] = data.loc[:, 'BldPitch1'] * (np.pi / 180) # convert to radians
        
        # replace spikes with average of surrounding average
        # Run remove_spikes for parameter sweep
        RUN_SPIKE_REMOVAL_PARAM_SWEEP = False
        if RUN_SPIKE_REMOVAL_PARAM_SWEEP:
            wl_vals = [8]
            po_vals = [1]
            thr_vals = [0.025]
            n_iter_vals = [4]
            pool = mp.Pool(mp.cpu_count())
            param_sweep = list(itertools.product(wl_vals, po_vals, thr_vals, n_iter_vals))
            time_data = data.loc[:, 'Time'].to_numpy()
            signal_data = data.loc[:, 'BldPitch1'].to_numpy()
            results = pool.starmap(test_filt_params, [(idx, params, signal_data, time_data) for idx, params in enumerate(param_sweep)])
            max_peak_vals = [res[1] for res in results]
            
            # print the parameters that are producing outlying peaks
            max_peak_mean = np.mean(max_peak_vals)
            max_peak_std = np.std(max_peak_vals)
            max_peak_zscore = (max_peak_vals - max_peak_mean) / max_peak_std
            candidate_params = []
            candidate_indices = []
            for i, z in enumerate(max_peak_zscore):
                if abs(z) < 3:
                    params = param_sweep[i]
                    candidate_indices.append(i)
                    candidate_params.append(params)
                    print(f'Case {i}: (window_length {params[0]}, polyorder {params[1]}, '
                          f'threshold {params[2]}, n_iter {params[3]}) is not an outlier')
            
            candidate_wl_vals = np.unique([par[0] for par in candidate_params])
            candidate_po_vals = np.unique([par[1] for par in candidate_params])
            candidate_thr_vals = np.unique([par[2] for par in candidate_params])
            candidate_n_iter_vals = np.unique([par[3] for par in candidate_params])
            print('window_length', candidate_wl_vals)
            print('polyorder', candidate_po_vals)
            print('threshold', candidate_thr_vals)
            print('n_iter', candidate_n_iter_vals)
            print(len(candidate_indices))

            
            n_rows = len(candidate_indices)
            n_plots = n_rows**2
            for p_idx, params in zip(candidate_indices[:min(n_plots, len(candidate_indices))],
                                    candidate_params[:min(n_plots, len(candidate_indices))]):
                fd = SmoothedFiniteDifference(smoother_kws={'window_length': params[0], 'polyorder': params[1]})
                data_filt = fd._differentiate(signal_data, t=time_data)
                data_filt = remove_spikes(data_filt[1:],
                                          time=time_data[1:],
                                          thresh=params[2], n_iter=params[3])
                fig, ax = plt.subplots(1, 1)
                ax.plot(time_data[1:], data_filt)
                ax.set_title(f'Case {p_idx}')
                plt.show()

        else:
    
            
            dt = data.loc[1, 'Time'] - data.loc[0, 'Time']
            idx = slice(init_idx + 500, init_idx + 1000, 1)
            
            # apply finite difference differentiation
            fd = FiniteDifference(d=1, order=2)
            fdd = FiniteDifference(d=2, order=2)
            
            data[f'BldPitch1_dot'] = fd._differentiate(data.loc[:, 'BldPitch1'].to_numpy(), t=dt)
            data[f'BldPitch1_ddot'] = fdd._differentiate(data.loc[:, 'BldPitch1'].to_numpy(), t=dt)

            # fig, ax = plt.subplots(3, 1)
            # ax[0].plot(data.loc[idx, 'Time'], data.loc[idx, 'BldPitch1'])
            # ax[1].plot(data.loc[idx, 'Time'], data.loc[idx, 'BldPitch1_dot'])
            # ax[2].plot(data.loc[idx, 'Time'], data.loc[idx, 'BldPitch1_ddot'])
            # plt.show()


        if use_filt:
            # fig, ax = plt.subplots(2, 1)
            # dt = data.loc[1, 'Time'] - data.loc[0, 'Time']
            # idx = slice(init_idx, len(data.index), 1)
            # ax[0].plot(data.loc[idx, 'Time'], data.loc[idx, 'Wind1VelX'])
            data.loc[:, 'Wind1VelX'] = generate_secord_lpf_var(data['Wind1VelX'])
            # ax[1].plot(data.loc[idx, 'Time'], data.loc[idx, 'Wind1VelX'])
            # plt.show()

        # pre-truncate data
        
        data.loc[:, 'P_motor'] = (data.loc[:, 'RootMzc1'] * data.loc[:, 'BldPitch1_dot'] * 1e-3).abs() # MW
        # data.loc[:, 'Azimuth'] = data.loc[:, 'Azimuth'] * (np.pi / 180)  # convert to radians
        fd = FiniteDifference(d=1, order=2)
        data.loc[:, 'Wind1VelX_dev_rated'] = data.loc[:, 'Wind1VelX'] - self.V_rated
        data.loc[:, 'Wind1VelX_dev_mean'] = data.loc[:, 'Wind1VelX'] - self.vmean_list[v_idx]
        data.loc[:, 'Wind1VelX_dev_mean_dot'] = fd._differentiate(data.loc[:, 'Wind1VelX_dev_mean'].to_numpy(), t=dt)
       
        data.drop(columns=['RootMzc1'], inplace=True)
        
        # data.loc[:, 'Wind1VelX_dev'] = data.loc[:, 'Wind1VelX'] - self.vmean_list[v_idx]

        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(data.loc[idx, 'Time'], data.loc[idx, 'Wind1VelX_dev_rated'])
        # ax[1].plot(data.loc[idx, 'Time'], data.loc[idx, 'Wind1VelX_dev_rated_dot'])
        # plt.show()

        data = data.iloc[init_idx:]


        data.attrs['seed'] = self.seed_list[s_idx]
        data.attrs['vmean'] = self.vmean_list[v_idx]

        return data

    def load_all_datasets(self, data_dir, use_filt=True, data_suffix='.SFunc.out',
                          data_prefix='AD_SOAR_c7_V2f_c73_Clean_',
                          outlist_path='/Users/aoifework/Documents/Research/ipc_tuning/plant_setup_package/OutList.mat', parallel=True, case_idx_inc=0):


        # BldPitch1, BldPitch1_dot and BldPitch1_ddot computed from filtered BldPitch1 model

        data_filenames = []
        vmean_indices = []
        seed_indices = []

        for path, _, files in os.walk(data_dir):
            for file in files:
                fsplit = file.split('.')
                fname = fsplit[0]
                ext = fsplit[-1]
                if ext == data_suffix and data_prefix in fname:
                    data_filenames.append(os.path.join(path, file))
                    
                    case_idx = int(fname.split('_')[-1]) - 1
                    vmean_indices.append(case_idx // len(self.seed_list))
                    seed_indices.append(case_idx % len(self.seed_list))

        # for each mean wind speed
        # for v_idx in range(len(self.vmean_list)):
        #     # for each turbulence seed
        #     for s_idx in range(len(self.seed_list)):
        #
        #         case_idx = (v_idx * s_idx) + s_idx + case_idx_inc
        #         vmean_indices.append(v_idx)
        #         seed_indices.append(s_idx)
        #
        #         # load the data
        #         if data_suffix == '.mat':
        #             data_filenames.append(os.path.join(data_dir, f'B_{self.vmean_list[v_idx]}_{self.seed_list[s_idx]}.mat'))
        #
        #         elif data_suffix == '.SFunc.out':
        #             data_filenames.append(os.path.join(data_dir, f'{data_prefix}{case_idx + 1}{data_suffix}'))

        # results as list of dicts
        if parallel:
            pool = Pool(processes=mp.cpu_count())
            self.data_all = pool.starmap(self.load_dataset, [(fn, v_idx, s_idx, data_suffix, use_filt, outlist_path)
                                                         for fn, v_idx, s_idx in zip(data_filenames, vmean_indices,
                                                                                     seed_indices)])
            pool.close()
        else:
            self.data_all = []
            for (case_idx, fn), v_idx, s_idx in zip(enumerate(data_filenames), vmean_indices, seed_indices):
                self.data_all.append(self.load_dataset(fn, v_idx, s_idx, data_suffix, use_filt))

        # convert to dict of lists

    def split_all_datasets(self, proportion_training_data, testing_idx=None):
        # Split into Training and Testing Datasets
        # n_datasets = len(self.data_all_normalized)
        self.n_training_datasets = round(proportion_training_data * self.n_datasets)
        self.n_testing_datasets = self.n_datasets - self.n_training_datasets
        # all_idx = np.arange(n_datasets)
        if testing_idx is None:
            testing_idx = []

        self.training_idx = set()
        while len(self.training_idx) < self.n_training_datasets:
            new_idx = np.floor(np.random.uniform(low=0, high=self.n_datasets, size=1)).astype(int)[0]
            if new_idx not in testing_idx:
                self.training_idx.add(new_idx)
        self.training_idx = np.array(list(self.training_idx))

        self.testing_idx = np.array([i for i in range(self.n_datasets) if i not in self.training_idx])
        # self.training_idx = np.arange(self.n_training_datasets)
        # self.testing_idx = np.arange(self.n_training_datasets, self.n_datasets) # all_idx[[i for i in all_idx if i not in training_idx]]

        self.data_all_split = defaultdict(list)
        self.P_motor_stats = defaultdict(list)

        self.data_all_split['input_data'] = self.data_all

        # for data_df in datasets:
        self.P_motor_stats[f'mean_seed'] = [data_df['P_motor'].mean() for data_df in self.data_all_split['input_data']]
        self.P_motor_stats[f'max_seed'] = [data_df['P_motor'].max() for data_df in self.data_all_split['input_data']]

        self.P_motor_stats['mean_vmean'] = [np.mean([data_df['P_motor'] for data_df in self.data_all_split['input_data']
                                                              if (data_df.attrs['vmean'] == vmean)])
                                                     for vmean in self.vmean_list]
        self.P_motor_stats['max_vmean'] = [np.max([data_df['P_motor'] for data_df in self.data_all_split['input_data']
                                                              if (data_df.attrs['vmean'] == vmean)])
                                                     for vmean in self.vmean_list]

    def load_data_all(self):

        for var_name in ['data_all', 'n_datasets']:
                        # + (['data_all'] if load_datasets else []):
            with open(f'./{var_name}.txt', 'rb') as fh:
                setattr(self, var_name, pickle.load(fh))

    def load_data_all_standardized(self):

        for var_name in ['data_all_standardized', 'scaler_std_', 'scaler_mean_']:
            with open(f'./{var_name}.txt', 'rb') as fh:
                setattr(self, var_name, pickle.load(fh))

    def save_data_all(self):

        for var_name in ['data_all', 'n_datasets']:
            with open(f'./{var_name}.txt', 'wb') as fh:
                pickle.dump(getattr(self, var_name), fh)

    def save_data_all_standardized(self):

        for var_name in ['data_all_standardized', 'scaler_std_', 'scaler_mean_']:
            with open(f'./{var_name}.txt', 'wb') as fh:
                pickle.dump(getattr(self, var_name), fh)

    def save_processed_data(self):

        for var_name in ['t_train', 't_test', 'u_train', 'u_test', 'x_train', 'x_dot_train', 'x_test', 'x_dot_test',
                         'feature_cols',
                         'state_cols', 'state_drvt_cols', 'ctrl_inpt_cols', 'n_features', 'vmean', 'vmean',
                         'seed', 'seed', 'n_training_datasets', 'n_testing_datasets',
                         'training_idx', 'testing_idx', 'P_motor_stats']:
            with open(f'./{var_name}.txt', 'wb') as fh:
                pickle.dump(getattr(self, var_name), fh)

    def load_processed_data(self):
        for var_name in ['t_train', 't_test', 'u_train', 'u_test', 'x_train', 'x_dot_train', 'x_test', 'x_dot_test', 'feature_cols',
                         'state_cols', 'state_drvt_cols', 'ctrl_inpt_cols', 'n_features', 'vmean', 'vmean',
                         'seed', 'seed', 'n_training_datasets', 'n_testing_datasets',
                         'training_idx', 'testing_idx', 'P_motor_stats']:
            with open(f'./{var_name}.txt', 'rb') as fh:
                setattr(self, var_name, pickle.load(fh))

    def process_data(self):

        dataset_train_df = [self.data_all_split['input_data'][i] for i in self.training_idx]
        dataset_test_df = [self.data_all_split['input_data'][i] for i in self.testing_idx]

        # drop channels that are algebraic functions of others
        # (dataset_train_df[0]['TwrBsM'] - (
        #             dataset_train_df[0]['TwrBsMxt'] ** 2 + dataset_train_df[0]['TwrBsMyt'] ** 2 ) ** (0.5)).std() => no corr
        # (dataset_train_df[0]['YawBrM'] - (
        #         dataset_train_df[0]['YawBrMxp'] ** 2 + dataset_train_df[0]['YawBrMyp'] ** 2  + dataset_train_df[0]['YawBrMzp'] ** 2) ** (0.5)).mean()  => no corr
        # (dataset_train_df[0]['RootMyb_Mag'] - (dataset_train_df[0]['RootMyb1']**2 + dataset_train_df[0]['RootMxb1']**2)**(0.5)).mean()
        # (dataset_train_df[0]['RootMyc0'] - (dataset_train_df[0]['RootMyc1']**2 + dataset_train_df[0]['RootMxc1']**2 + dataset_train_df[0]['RootMzc1']**2)**(0.5)).mean() => not correlated
        # (dataset_train_df[0]['YawBrM'] - (dataset_train_df[0]['YawBrMzp']**2 + dataset_train_df[0]['YawBrMyp']**2 + dataset_train_df[0]['YawBrMxp']**2)**(0.5)).mean() => not correlated

        # (dataset_train_df[0]['TSR'] / (dataset_train_df[0]['RotSpeed'] / dataset_train_df[0]['Wind1VelX'])).std() ~ 0 => TSR = RotSpeed/Wind1VelX * Radius
        # (dataset_train_df[0]['GenSpeed']/dataset_train_df[0]['RotSpeed']).std() = 0 => GenSpeed = RotSpeed
        # (dataset_train_df[0]['RtTSR'] - dataset_train_df[0]['TSR']).std() => not sure, try plotting
        # (dataset_train_df[0]['GenTq'] / dataset_train_df[0]['RotTorq']).std() ~ 0 => GenTq = RotTq
        # (dataset_train_df[0]['GenPwr'] / (dataset_train_df[0]['RotSpeed'] * dataset_train_df[0]['GenTq'])).std() ~ 0 => GenPwr = GenSpeed * GenTq
        # (dataset_train_df[0]['RotThrust'] / dataset_train_df[0]['RotTorq']).std()
        # plt.scatter(dataset_train_df[0]['RotThrust'], dataset_train_df[0]['RotTorq'])
        # plt.show()

        self.ctrl_inpt_cols = {
            'P_motor_partial': ['BldPitch1', 'BldPitch1_dot', 'BldPitch1_ddot']
        }


        self.state_cols = {
            'P_motor_partial': ['P_motor']
        }

        if self.model_per_wind_speed:
            self.ctrl_inpt_cols = {**self.ctrl_inpt_cols,
                              **{f'BldPitch1_{v}': [c for c in SUBSET_MODEL_CHANNELS
                                                    if c not in ['Time', 'BldPitch1', 'P_motor', 'BldPitch1_dot', 'BldPitch1_ddot']]
                                 for v in set(self.vmean_list)}}

            self.state_cols = {**self.state_cols,
                          **{f'BldPitch1_{v}': [c for c in SUBSET_MODEL_CHANNELS
                                                if c not in self.ctrl_inpt_cols[f'BldPitch1_{v}']
                                                + ['P_motor', 'Time', 'BldPitch1_dot']]
                             for v in set(self.vmean_list)}}

        else:
            self.ctrl_inpt_cols = {**self.ctrl_inpt_cols,
                              **{f'BldPitch1': [c for c in SUBSET_MODEL_CHANNELS
                                                if c not in ['Time', 'BldPitch1', 'P_motor', 'BldPitch1_dot',
                                                             'BldPitch1_ddot']]}}

            include_drvts_in_model = False

            self.state_cols = {**self.state_cols,
                          **{f'BldPitch1': [c for c in SUBSET_MODEL_CHANNELS
                                            if c not in self.ctrl_inpt_cols['BldPitch1']
                                            + ['P_motor', 'Time', 'BldPitch1_ddot']
                                            + ([] if include_drvts_in_model else ['BldPitch1_dot'])]}}

            self.state_drvt_cols = {
                'P_motor_partial': [],
                'BldPitch1': ['BldPitch1_dot']  # , 'BldPitch1_ddot']
            }


        self.t_train = dataset_train_df[0]['Time'].to_numpy()
        self.t_test = dataset_test_df[0]['Time'].to_numpy()
        self.t_full = np.concatenate([self.t_train, self.t_test])

        self.u_train = {key: [dataset_train_df[i].loc[:, self.ctrl_inpt_cols[key]].to_numpy()
                         for i in range(self.n_training_datasets)]
                   for key in self.model_names}
        self.u_test = {key: [dataset_test_df[i].loc[:, self.ctrl_inpt_cols[key]].to_numpy()
                        for i in range(self.n_testing_datasets)]
                  for key in self.model_names}

        self.x_train = {key: [dataset_train_df[i].loc[:, self.state_cols[key]].to_numpy()
                         for i in range(self.n_training_datasets)]
                   for key in self.model_names}

        self.x_test = {key: [dataset_test_df[i].loc[:, self.state_cols[key]].to_numpy()
                        for i in range(self.n_testing_datasets)]
                  for key in self.model_names}

        self.x_dot_train = {key: [dataset_train_df[i].loc[:, self.state_drvt_cols[key]].to_numpy()
                                  for i in range(self.n_training_datasets)]
                            for key in self.model_names}

        self.x_dot_test = {key: [dataset_test_df[i].loc[:, self.state_drvt_cols[key]].to_numpy()
                                  for i in range(self.n_testing_datasets)]
                            for key in self.model_names}

        self.feature_cols = {key: self.state_cols[key] + self.ctrl_inpt_cols[key] for key in self.model_names}
        self.n_features = {key: len(self.feature_cols[key]) for key in self.model_names}
        self.vmean = {'train': [dataset_train_df[i].attrs['vmean'] for i in range(self.n_training_datasets)],
                      'test': [dataset_test_df[i].attrs['vmean'] for i in range(self.n_testing_datasets)]}
        self.seed = {'train': [dataset_train_df[i].attrs['seed'] for i in range(self.n_training_datasets)],
                     'test': [dataset_test_df[i].attrs['seed'] for i in range(self.n_testing_datasets)]}

        del dataset_train_df, dataset_test_df


    def generate_candidate_funcs(self):
        
        all_libs = {key: [] for key in self.model_names}
        inputs_per_library = {key: np.zeros((0, self.n_features[key]), dtype=int) for key in self.model_names}

        # if any('BldPitch1' in key for key in self.model_names):
        for key in self.model_names:
            if 'BldPitch1' in key:

                
                ## POLYNOMIAL LIBRARY
                poly_lib2 = PolynomialLibrary(degree=3, include_interaction=True, include_bias=True)

                v_rated_idx = [self.feature_cols[key].index('Wind1VelX_dev_rated')]
                v_idx = [self.feature_cols[key].index('Wind1VelX_dev_mean')]
                v_dot_idx = [self.feature_cols[key].index('Wind1VelX_dev_mean_dot')]
                beta_idx = [self.feature_cols[key].index('BldPitch1')]

                all_libs[key].append(poly_lib2)
                inputs_per_library[key] = np.vstack([inputs_per_library[key],
                                                     ((v_idx + v_dot_idx) * self.n_features[key])[
                                                         :self.n_features[key]
                                                     ]])

                ## POLYNOMIAL ROOT LIBRARY
                # poly_lib1_funcs = [poly_lib1_func1]#, poly_lib1_func2] #, poly_lib1_func3]
                # poly_lib1_func_names = [poly_lib1_funcname1]#, poly_lib1_funcname2] #, poly_lib1_funcname3]
                # poly_lib1 = CustomLibrary(library_functions=poly_lib1_funcs, function_names=poly_lib1_func_names)

                # all_libs[key].append(poly_lib1)
                # inputs_per_library[key] = np.vstack([inputs_per_library[key],
                #                                      ((v_rated_idx) * self.n_features[key])[
                #                                      :self.n_features[key]]])

                ## TENSORED LIBRARY
                # poly_lib3 = PolynomialLibrary(degree=1, include_interaction=False, include_bias=True)
                # tensored_lib = poly_lib1 * poly_lib3
                # all_libs[key].append(tensored_lib)
                # inputs_per_library[key] = np.vstack([inputs_per_library[key],
                #                                      ((v_idx + v_dot_idx) * self.n_features[key])[
                #                                       :self.n_features[key]
                #                                      ]])
                
                ## SINUSOIDAL LIBRARY
                # fourier_lib = FourierLibrary(include_sin=True, include_cos=True, n_frequencies=3)
                #
                # all_libs[key].append(fourier_lib)
                # inputs_per_library[key] = np.vstack([inputs_per_library[key],
                #                                      # (sinus_indices
                #                                      (v_idx
                #                                       * self.n_features[key])[
                #                                      :self.n_features[key]]])
                
                

            elif key == 'P_motor_partial':
                beta_idx = [self.feature_cols['P_motor_partial'].index('BldPitch1')]
                beta_dot_idx = [self.feature_cols['P_motor_partial'].index('BldPitch1_dot')]
                beta_ddot_idx = [self.feature_cols['P_motor_partial'].index('BldPitch1_ddot')]
                pmotor_idx = [self.feature_cols['P_motor_partial'].index('P_motor')]

                inc_indices_0 = beta_idx + beta_dot_idx + beta_ddot_idx# + pmotor_idx
                all_libs['P_motor_partial'].append(PolynomialLibrary(degree=2, include_interaction=True, include_bias=False))
                inputs_per_library['P_motor_partial'] = np.vstack([inputs_per_library['P_motor_partial'],
                                                                   (inc_indices_0
                                                                    * self.n_features['P_motor_partial'])[
                                                                   :self.n_features['P_motor_partial']]])


        # Join libraries
        self.full_lib = {key: GeneralizedLibrary(all_libs[key],
                                            inputs_per_library=inputs_per_library[key])
                    for key in self.model_names}

def poly_lib1_func1(x):
    return (np.maximum(np.zeros_like(x), x)) ** (1 / 2)

def poly_lib1_func2(x):
    return (np.maximum(np.zeros_like(x), x)) ** (1 / 3)

def poly_lib1_func3(x):
    # return np.maximum(np.zeros_like(x), x)
    try:
        return np.log10(np.abs(x))
    except Exception as e:
        print('oh no')

def poly_lib1_funcname1(x):
    return f"max(0, {x})^(1/2)"

def poly_lib1_funcname2(x):
    return f"max(0, {x})^(1/3)"

def poly_lib1_funcname3(x):
    # return f"max(0, {x})"
    return f"log_10({x})"


def test_filt_params(p_idx, params, signal_data, time_data):
    print(f'Case {p_idx}: window_length {params[0]}, polyorder {min(params[1], params[0] - 1)}, '
          f'threshold {params[2]}, n_iter {params[3]}')
    fd = SmoothedFiniteDifference(smoother_kws={'window_length': params[0], 'polyorder': min(params[1], params[0] - 1)})
    data = fd._differentiate(signal_data, t=time_data)
    data = remove_spikes(data[1:], time=time_data[1:],thresh=params[2], n_iter=params[3])
    
    return p_idx, max(data)