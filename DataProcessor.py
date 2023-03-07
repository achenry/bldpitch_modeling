import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from pysindy.feature_library import CustomLibrary, PolynomialLibrary, GeneralizedLibrary, FourierLibrary
import os
from sklearn.linear_model._base import safe_sparse_dot
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from pysindy import FiniteDifference
from scipy.signal import lfilter, lti
import pickle


def generate_bldpitch1_true_drvts(case_idx, data_processor):
    t_full = np.concatenate([data_processor.t_train, data_processor.t_test])
    return {
        'time': t_full,
        'BldPitch1': np.concatenate(
            [data_processor.u_train['P_motor'][case_idx][:, data_processor.ctrl_inpt_cols['P_motor'].index('BldPitch1')],
             data_processor.u_test['P_motor'][case_idx][:, data_processor.ctrl_inpt_cols['P_motor'].index('BldPitch1')]]),
        'BldPitch1_dot': np.concatenate(
            [data_processor.u_train['P_motor'][case_idx][:, data_processor.ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')],
             data_processor.u_test['P_motor'][case_idx][:, data_processor.ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')]]),
        'BldPitch1_ddot': np.concatenate(
            [data_processor.u_train['P_motor'][case_idx][:, data_processor.ctrl_inpt_cols['P_motor'].index('BldPitch1_ddot')],
             data_processor.u_test['P_motor'][case_idx][:, data_processor.ctrl_inpt_cols['P_motor'].index('BldPitch1_ddot')]])
    }

def compute_mse(data_true, data_modeled):
    n_datasets = len(data_true)
    return (1 / n_datasets) * np.sum([(y_true - y_mod) ** 2
                                      for y_true, y_mod in zip(data_true, data_modeled)])


def butter_lowpass(cutoff, ts, order=5):
    nyq = 0.5 * (1 / ts)
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, ts, order=5):
    b, a = butter_lowpass(cutoff, ts, order=order)
    y = lfilter(b, a, data)
    return y

class DataProcessor:
    def __init__(self, **kwargs):
        self.vmean_list = kwargs['vmean_list']
        self.seed_list = kwargs['seed_list']
        self.V_rated = kwargs['V_rated']
        self.GenSpeed_rated = kwargs['GenSpeed_rated']

        self.data_all = {'P_motor_mean_seed': [], 'P_motor_mean_vmean': [],
                         'P_motor_max_seed': [], 'P_motor_max_vmean': [],
                         'seed': [], 'vmean': [], 'filename': [], 'dt': [],
                         'training_input_data': [], 'testing_input_data': [],
                         'training_output_data': [], 'testing_output_data': [],
                         'n_datapoints': []}


        self.n_wind = len(self.vmean_list)
        self.n_seed = len(self.seed_list)
        self.n_datasets = self.n_wind * self.n_seed
        self.model_names = kwargs['model_names']
        self.model_per_wind_speed = kwargs['model_per_wind_speed']


    def load_data(self, data_dir):

        # for each mean wind speed
        for v_idx in range(len(self.vmean_list)):
            # for each turbulence seed
            for s_idx in range(len(self.seed_list)):

                # load the data
                data_filename = os.path.join(data_dir, f'B_{self.vmean_list[v_idx]}_{self.seed_list[s_idx]}.mat')
                print(f'> loading {data_filename} ..\n')
                data = loadmat(data_filename)

                # extract the output channel names
                if v_idx == 0 and s_idx == 0:
                    op_channels = [ch[0] for ch in data['Chan'][0].dtype.descr]

                # create a dictionary of the data
                data_dict = {ch_name: data['Chan'][0][0][ch_idx] for ch_idx, ch_name in enumerate(op_channels)}

                # generate beta_dot and beta_ddot values
                init_idx = 5001
                dt = (data_dict['tt'][1] - data_dict['tt'][0])[0]
                t = data_dict['tt'][init_idx:].squeeze()

                # beta_dot1 = [
                #     (data_dict[f'BldPitch{b + 1}'][init_idx + 1:] - data_dict[f'BldPitch{b + 1}'][init_idx - 1:-2]) \
                #     / (2 * dt) for b in range(3)]
                # beta_ddot1 = [
                #     (data_dict[f'BldPitch{b + 1}'][init_idx + 1:] - 2 * data_dict[f'BldPitch{b + 1}'][init_idx:-1]
                #      + data_dict[f'BldPitch{b + 1}'][init_idx - 1:-2]) / (2 * dt) for b in range(3)]

                fd = FiniteDifference()
                data_dict['BldPitch1'] = data_dict['BldPitch1'] * np.pi / 180
                beta = data_dict[f'BldPitch1'][init_idx:].squeeze()
                beta_dot = fd._differentiate(beta, t=t)
                beta_ddot = fd._differentiate(beta_dot, t=t)

                # pre-truncate data
                for k, v in data_dict.items():
                    data_dict[k] = v[init_idx:]

                # for b in range(3):
                #     data_dict[f'BldPitch{b+1}_dot'] = beta_dot[b]
                #     data_dict[f'BldPitch{b+1}_ddot'] = beta_ddot[b]

                # GENERATE NEW VARIABLES (scaled or composition of other vars)
                data_dict[f'BldPitch1_dot'] = beta_dot
                data_dict[f'BldPitch1_ddot'] = beta_ddot

                data_dict['RootMzc1'] = data_dict['RootMzc1'] * 1e-4

                P_motor = np.multiply(data_dict['RootMzc1'], data_dict['BldPitch1_dot'][:, np.newaxis])
                data_dict['P_motor'] = P_motor

                data_dict['Azimuth'] = data_dict['Azimuth'] * (np.pi / 180)
                data_dict['GenTq'] = data_dict['GenTq'] * 1e-4

                data_dict['GenSpeed'] = data_dict['GenSpeed'] * (2 * np.pi / 60) * 100

                # data_dict['Wind1VelX_filt'] = generate_secord_lpf_var(data_dict['Wind1VelX'])
                # data_dict['GenSpeed_filt'] = generate_firstord_lpf_var(data_dict['GenSpeed']) * (2 * np.pi / 60) * 100

                # data_dict['Wind1VelX_filt_dev_rated'] = data_dict['Wind1VelX_filt'] - self.V_rated
                # data_dict['GenSpeed_filt_dev_rated'] = data_dict['GenSpeed_filt'] - (self.GenSpeed_rated * 100)
                # data_dict['Wind1VelX_dev_filt'] = data_dict['Wind1VelX_filt'] - self.vmean_list[v_idx]

                data_dict['Wind1VelX_dev_rated'] = data_dict['Wind1VelX'] - self.V_rated
                data_dict['GenSpeed_dev_rated'] = data_dict['GenSpeed'] - (self.GenSpeed_rated * 100)
                data_dict['Wind1VelX_dev'] = data_dict['Wind1VelX'] - self.vmean_list[v_idx]

                # del data_dict['GenSpeed']
                # del data_dict['Wind1VelX']

                # squeeze data
                for k, v in data_dict.items():
                    data_dict[k] = np.squeeze(v)

                n_datapoints = len(data_dict['tt'])

                # compute true bladepitch motor power

                P_motor_mean_seed = np.mean(P_motor)
                P_motor_max_seed = (np.max(np.abs(P_motor)) * (
                    P_motor[np.argmax(np.abs(P_motor))] / np.abs(P_motor[np.argmax(np.abs(P_motor))])))[0]
                if np.max(np.abs(P_motor)) == 0:
                    P_motor_max_seed = 0.0
                # data_dict_output = {}
                # data_dict_output['P_motor'] = P_motor
                # data_dict_output['BldPitch1'] = data_dict['BldPitch1']

                # add parameters of this dataset
                # self.data_all['Wind1VelX_mean'].append(self.vmean_list[v_idx])
                # self.data_all['GenSpeed_mean'].append(np.mean(data_dict['GenSpeed']))
                self.data_all['seed'].append(self.seed_list[s_idx])
                self.data_all['vmean'].append(self.vmean_list[v_idx])
                self.data_all['dt'].append(dt)
                # self.data_all['Wind1VelX_std'].append(np.std(data_dict['Wind1VelX']))
                self.data_all['filename'].append(data_filename)
                self.data_all['P_motor_mean_seed'].append(P_motor_mean_seed)
                self.data_all['P_motor_max_seed'].append(P_motor_max_seed)
                self.data_all['n_datapoints'].append(n_datapoints)

                # Split into Training and Testing Data
                n_training = int(0.8 * n_datapoints)
                all_idx = np.arange(n_datapoints)
                # training_idx = np.sort(np.floor(np.random.uniform(low=0, high=n_datapoints, size=n_training)).astype(int))
                training_idx = np.arange(n_training)
                testing_idx = all_idx[[i for i in all_idx if i not in training_idx]]
                n_testing = n_datapoints - n_training

                # add input datasets
                data_dict_training = {}
                data_dict_testing = {}
                for k, v in data_dict.items():
                    data_dict_training[k] = v[training_idx]
                    data_dict_testing[k] = v[testing_idx]
                self.data_all['training_input_data'].append(data_dict_training)
                self.data_all['testing_input_data'].append(data_dict_testing)

                # add output datasets
                # data_dict_training_output = {}
                # data_dict_testing_output = {}
                # for k, v in data_dict_output.items():
                #     data_dict_training_output[k] = v[training_idx]
                #     data_dict_testing_output[k] = v[testing_idx]
                # self.data_all['training_output_data'].append(data_dict_training_output)
                # self.data_all['testing_output_data'].append(data_dict_testing_output)

            P_motor_mean_vmean = np.mean(
                [P_val for vmean, P_val in zip(self.data_all['vmean'], self.data_all['P_motor_mean_seed'])
                 if vmean == self.vmean_list[v_idx]])
            P_motor_max_vmean = np.mean(
                [P_val for vmean, P_val in zip(self.data_all['vmean'], self.data_all['P_motor_max_seed'])
                 if vmean == self.vmean_list[v_idx]])
            self.data_all['P_motor_mean_vmean'] = self.data_all['P_motor_mean_vmean'] + ([P_motor_mean_vmean] * self.n_seed)
            self.data_all['P_motor_max_vmean'] = self.data_all['P_motor_max_vmean'] + ([P_motor_max_vmean] * self.n_seed)


    def load_data_all(self):
        with open(f'./data_all.txt', 'rb') as fh:
            self.data_all = pickle.load(fh)

    def save_data_all(self):
        with open(f'./data_all.txt', 'wb') as fh:
            pickle.dump(self.data_all, fh)

    def save_processed_data(self):

        for var_name in ['t_train', 't_test', 'u_train', 'u_test', 'x_train', 'x_dot_train', 'x_test', 'feature_cols',
                         'state_cols', 'state_drvt_cols', 'ctrl_inpt_cols', 'n_features']:
            with open(f'./{var_name}.txt', 'wb') as fh:
                pickle.dump(getattr(self, var_name), fh)

    def load_processed_data(self):
        for var_name in ['t_train', 't_test', 'u_train', 'u_test', 'x_train', 'x_dot_train', 'x_test', 'feature_cols',
                         'state_cols', 'state_drvt_cols', 'ctrl_inpt_cols', 'n_features']:
            with open(f'./{var_name}.txt', 'rb') as fh:
                setattr(self, var_name, pickle.load(fh))

    def process_data(self):
    
        ## Export Datasets to Arrays
        # V_mean = [12, 14, 20]
        # dataset_indices = [i for i in range(len(self.data_all['Wind1VelX_mean'])) if self.data_all['Wind1VelX_mean'][i] in V_mean]

        dataset_indices = range(self.n_datasets)

        # Note that GenTq and RotSpeed candidate funcs for poly degree 2are algebraic functions of eachother
        # BldPitch1, RootMzc1, Wind1VelX, Wind1VelX_dev, Wind1VelX_dev_rated, GenTq, GenSpeed, GenSpeed_dev_rated, OoPDefl1, IPDefl1, Azimuth
        subset_in_channels = [
            'tt',
            # 'RootMyc0', #'RootMyc1','RootMxc1',
            # 'RootMzc1',
            # 'RootMyb_Mag', #'RootMxb1', 'RootMyb1',
            # 'TipBendM',
            # 'TSR', 'RtTSR',
            'Wind1VelX',
            # 'Wind1VelX_filt',
            'GenTq',
            # 'RtAeroCp',
            # 'RotThrust',
            # 'RotSpeed',
            'GenSpeed',
            # 'GenSpeed_filt',
            'OoPDefl1', 'IPDefl1',
            # 'TwrBsM', #'TwrBsMyt', 'TwrBsMxt',
            # 'TwrClrnc1',
            # 'TTDspFA', 'TTDspSS',
            # 'YawBrM', #'YawBrMzp', 'YawBrMyp', 'YawBrMxp',
            # 'Wind1VelY', 'Wind1VelZ',
            'Azimuth',
            'BldPitch1', 'BldPitch1_dot',  # 'BldPitch1_ddot',
            'P_motor',
            # 'Wind1VelX_filt_dev', 'Wind1VelX_filt_dev_rated', 'GenSpeed_filt_dev_rated'
            'Wind1VelX_dev', 'Wind1VelX_dev_rated', 'GenSpeed_dev_rated'
        ]

        dataset_train_df = [pd.DataFrame(self.data_all['training_input_data'][i], columns=subset_in_channels)
                            for i in dataset_indices]
        dataset_test_df = [pd.DataFrame(self.data_all['testing_input_data'][i], columns=subset_in_channels)
                           for i in dataset_indices]

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
            'P_motor': [c for c in subset_in_channels
                        if c not in ['tt', 'P_motor']]
        }


        self.state_cols = {
            'P_motor': [c for c in subset_in_channels
                        if c not in self.ctrl_inpt_cols['P_motor']
                        + ['tt']]
        }

        if self.model_per_wind_speed:
            self.ctrl_inpt_cols = {**self.ctrl_inpt_cols,
                              **{f'BldPitch1_{v}': [c for c in subset_in_channels
                                                    if c not in ['tt', 'BldPitch1', 'P_motor', 'BldPitch1_dot', 'BldPitch1_ddot']]
                                 # 'BldPitch1_dot': [c for c in subset_in_channels
                                 #                   if c not in ['tt', 'BldPitch1', 'P_motor', 'BldPitch1_dot', 'BldPitch1_ddot']],
                                 # 'BldPitch1_ddot': [c for c in subset_in_channels
                                 #                    if c not in ['tt', 'BldPitch1', 'P_motor', 'BldPitch1_dot', 'BldPitch1_ddot']]
                                 for v in set(self.data_all['vmean'])}}

            self.state_cols = {**self.state_cols,
                          **{f'BldPitch1_{v}': [c for c in subset_in_channels
                                                if c not in self.ctrl_inpt_cols[f'BldPitch1_{v}']
                                                + ['P_motor', 'tt', 'BldPitch1_dot']]
                             # 'BldPitch1_dot': [c for c in subset_in_channels
                             #                   if c not in ['tt', 'BldPitch1', 'P_motor', 'BldPitch1_dot', 'BldPitch1_ddot']],
                             # 'BldPitch1_ddot': [c for c in subset_in_channels
                             #                    if c not in ['tt', 'BldPitch1', 'P_motor', 'BldPitch1_dot', 'BldPitch1_ddot']]
                             for v in set(self.data_all['vmean'])}}


            self.state_drvt_cols = {
                f'BldPitch1_{v}': [f'BldPitch1_dot']
                for v in set(self.data_all['vmean'])
            }


        else:
            self.ctrl_inpt_cols = {**self.ctrl_inpt_cols,
                              **{f'BldPitch1': [c for c in subset_in_channels
                                                if c not in ['tt', 'BldPitch1', 'P_motor', 'BldPitch1_dot',
                                                             'BldPitch1_ddot']]}}

            self.state_cols = {**self.state_cols,
                          **{f'BldPitch1': [c for c in subset_in_channels
                                            if c not in self.ctrl_inpt_cols['BldPitch1']
                                            + ['P_motor', 'tt', 'BldPitch1_dot']]}}

            self.state_drvt_cols = {
                'BldPitch1': ['BldPitch1_dot']  # , 'BldPitch1_ddot']
            }


        self.t_train = dataset_train_df[0]['tt'].to_numpy()
        self.t_test = dataset_test_df[0]['tt'].to_numpy()
        self.t_full = np.concatenate([self.t_train, self.t_test])

        self.u_train = {key: [dataset_train_df[i].loc[:, self.ctrl_inpt_cols[key]].to_numpy()
                         for i in range(len(dataset_indices))]
                   for key in self.model_names}
        self.u_test = {key: [dataset_test_df[i].loc[:, self.ctrl_inpt_cols[key]].to_numpy()
                        for i in range(len(dataset_indices))]
                  for key in self.model_names}

        self.x_train = {key: [dataset_train_df[i].loc[:, self.state_cols[key]].to_numpy()
                         for i in range(len(dataset_indices))]
                   for key in self.model_names}

        self.x_dot_train = {key: [dataset_train_df[i].loc[:, self.state_drvt_cols[key]].to_numpy()
                             for i in range(len(dataset_indices))]
                       for key in self.model_names}

        self.x_test = {key: [dataset_test_df[i].loc[:, self.state_cols[key]].to_numpy()
                        for i in range(len(dataset_indices))]
                  for key in self.model_names}

        self.feature_cols = {key: self.state_cols[key] + self.ctrl_inpt_cols[key] for key in self.model_names}
        self.n_features = {key: len(self.feature_cols[key]) for key in self.model_names}


    def generate_candidate_funcs(self):
        all_libs = {key: [] for key in self.model_names}
        inputs_per_library = {key: np.zeros((0, self.n_features[key]), dtype=int) for key in self.model_names}

        # TODO turbulence terms based on DLC12, NTM (? which class A, B or C), which spectrum

        # fraction polynomial
        poly_lib1_funcs = [
            lambda x: (np.maximum(np.zeros_like(x), x)) ** (1 / 2),
            lambda x: (np.maximum(np.zeros_like(x), x)) ** (1 / 3),
            lambda x: np.maximum(np.zeros_like(x), x)]

        poly_lib1_func_names = [lambda x: f"max(0, {x})^(1/2)",
                                lambda x: f"max(0, {x})^(1/3)",
                                lambda x: f"max(0, {x})"]

        poly_lib1 = {key: CustomLibrary(library_functions=poly_lib1_funcs, function_names=poly_lib1_func_names)
                     for key in self.model_names}

        # Polynomial of all but GenTq,
        # considering low-pass-filtered rotor speed and wind speed variables
        poly_lib2 = {key: PolynomialLibrary(degree=2, include_interaction=True, include_bias=True)
                     for key in self.model_names}

        if 'P_motor' in self.model_names:
            all_libs['P_motor'].append(poly_lib2['P_motor'])
            idx = np.array([1] + list(range(1, self.n_features['P_motor'])))
            idx[self.feature_cols['P_motor'].index('GenTq')] = self.feature_cols['P_motor'].index('Wind1VelX_filt')
            idx[self.feature_cols['P_motor'].index('GenSpeed_filt_dev_rated')] = self.feature_cols['P_motor'].index('Wind1VelX_filt')
            idx[self.feature_cols['P_motor'].index('GenSpeed_dev_rated')] = self.feature_cols['P_motor'].index('Wind1VelX_filt')
            idx[self.feature_cols['P_motor'].index('Wind1VelX_filt_dev_rated')] = self.feature_cols['P_motor'].index('Wind1VelX_filt')
            idx[self.feature_cols['P_motor'].index('Wind1VelX_dev_rated')] = self.feature_cols['P_motor'].index('Wind1VelX_filt')
            idx[self.feature_cols['P_motor'].index('Wind1VelX_filt_dev')] = self.feature_cols['P_motor'].index('Wind1VelX_filt')
            idx[self.feature_cols['P_motor'].index('Wind1VelX_dev')] = self.feature_cols['P_motor'].index('Wind1VelX_filt')
            # idx[self.feature_cols['P_motor'].index('Azimuth')] = self.feature_cols['P_motor'].index('Wind1VelX')
            inputs_per_library['P_motor'] = np.vstack([inputs_per_library['P_motor'], idx])

        if any('BldPitch1' in key for key in self.model_names):

            for key in self.model_names:
                if 'BldPitch1' in key:

                    base_idx = list(range(self.n_features[key]))
                    cut_idx = [self.feature_cols[key].index(f) for f in
                               ['IPDefl1', 'Wind1VelX_dev_rated', 'GenSpeed']]
                    # ['IPDefl1', 'Wind1VelX_filt_dev_rated', 'GenSpeed', 'GenSpeed_filt']]
                    in_idx = [i for i in base_idx if i not in cut_idx]
                    for i in cut_idx:
                        base_idx[i] = in_idx[0]

                    idx = np.array(base_idx)

                    all_libs[key].append(poly_lib2[key])
                    inputs_per_library[key] = np.vstack([inputs_per_library[key], idx])

                    # v_dev_rated_idx = [self.feature_cols[key].index('Wind1VelX_filt_dev_rated')]
                    v_dev_rated_idx = [self.feature_cols[key].index('Wind1VelX_dev_rated')]
                    # genspeed_dev_rated_idx = [self.feature_cols[key].index('GenSpeed_filt_dev_rated')]
                    genspeed_dev_rated_idx = [self.feature_cols[key].index('GenSpeed_dev_rated')]

                    all_libs[key].append(poly_lib1[key])
                    inputs_per_library[key] = np.vstack([inputs_per_library[key],
                                                         ((v_dev_rated_idx + genspeed_dev_rated_idx) * self.n_features[key])[
                                                         :self.n_features[key]]])

            ['BldPitch1',  # 'BldPitch1_dot',  # 'BldPitch1_ddot',
             'Wind1VelX',
             'GenTq', 'GenSpeed',
             'OoPDefl1', 'IPDefl1', 'Azimuth',
             'Wind1VelX_dev', 'Wind1VelX_dev_rated', 'GenSpeed_dev_rated']

            # idx[self.feature_cols['BldPitch1'].index('GenTq')] = self.feature_cols['BldPitch1'].index('Wind1VelX')
            # idx[self.feature_cols['BldPitch1'].index('GenSpeed_dev_rated')] = self.feature_cols['BldPitch1'].index('Wind1VelX')
            # idx[self.feature_cols['BldPitch1'].index('Wind1VelX_dev_rated')] = self.feature_cols['BldPitch1'].index('Wind1VelX')
            # idx[self.feature_cols['BldPitch1'].index('Wind1VelX_dev')] = self.feature_cols['BldPitch1'].index('Wind1VelX')
            # idx[self.feature_cols['BldPitch1'].index('Azimuth')] = self.feature_cols['BldPitch1'].index('Wind1VelX')

        # Sinusoidal theta terms - \sin(\theta), \cos(\theta), \sin(\theta)\cos(\theta), \sin^2(\theta)
        # model with only sin and cos - bad; with cossin and sin^2 - much better
        # sin_lib_funcs = [
        #     lambda x: np.abs(x) * np.sin(x),
        #     lambda x: np.abs(x) * np.sin(2 * x),
        #     lambda x: np.abs(x) * np.cos(x),
        #     lambda x: np.abs(x) * np.cos(2 * x),
        # ]
        # sin_lib_func_names = [
        #                       lambda x: f"{x}sin({x})",
        #                       lambda x: f"{x}sin(2{x})",
        #                       lambda x: f"{x}cos({x})",
        #                       lambda x: f"{x}cos(2{x})",
        #                       ]
        #
        # sin_lib = {key: CustomLibrary(library_functions=sin_lib_funcs, function_names=sin_lib_func_names) for key in model_names}

        poly_four_lib = PolynomialLibrary(degree=1, include_interaction=True, include_bias=False)
        fourier_lib = {key: poly_four_lib * FourierLibrary(include_sin=True, include_cos=True, n_frequencies=3) for key in
                       self.model_names}

        if 'P_motor' in self.model_names:
            all_libs['P_motor'].append(fourier_lib['P_motor'])
            inputs_per_library['P_motor'] = np.vstack([inputs_per_library['P_motor'],
                                                       [1] + list(range(1, self.n_features['P_motor']))])

        if any('BldPitch1' in key for key in self.model_names):

            for key in self.model_names:
                if 'BldPitch1' in key:
                    # pass # 1 deg 3 poly, 2 deg 3 poly w bias, 3 deg 3 poly w bias and fourier

                    all_libs[key].append(fourier_lib[key])
                    theta_idx = [self.feature_cols[key].index('Azimuth')]
                    # genspeed_idx = [self.feature_cols[key].index('GenSpeed_filt')]
                    genspeed_idx = [self.feature_cols[key].index('GenSpeed')]
                    tipdefl1_idx = [self.feature_cols[key].index('OoPDefl1')]
                    tipdefl2_idx = [self.feature_cols[key].index('IPDefl1')]
                    inputs_per_library[key] = np.vstack([inputs_per_library[key],
                                                         ((theta_idx + genspeed_idx + tipdefl1_idx + tipdefl2_idx) *
                                                          self.n_features[key])[
                                                         :self.n_features[key]]])

            # inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'],
            #                                              ([ctrl_inpt_idx] * ctrl_inpt_idx) + list(range(ctrl_inpt_idx, self.n_features['BldPitch1']))])

            # Beta_bar schedule terms: (max(0, V - Vrated))^1/2, (max(0, V - Vrated))^1/3, max(0, V - Vrated)
            # v_dev_rated_idx = {key: [self.feature_cols[key].index('Wind1VelX_dev_rated')]
            #                    for key in ['BldPitch1']}
            #
            # beta0_lib_funcs = [
            #     lambda x: (np.maximum(np.zeros_like(x), x)) ** (1 / 2),
            #     lambda x: (np.maximum(np.zeros_like(x), x)) ** (1 / 3),
            #     lambda x: np.maximum(np.zeros_like(x), x)]
            #
            # beta0_lib_func_names = [lambda x: f"max(0, {x})^(1/2)",
            #                         lambda x: f"max(0, {x})^(1/3)",
            #                         lambda x: f"max(0, {x})"]
            #
            # beta0_lib = {key: CustomLibrary(library_functions=beta0_lib_funcs, function_names=beta0_lib_func_names) for key in
            #              ['BldPitch1']}
            #
            # all_libs['BldPitch1'].append(beta0_lib['BldPitch1'])
            #
            # inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'],
            #                                              (v_dev_rated_idx['BldPitch1'] * self.n_features['BldPitch1'])[
            #                                              :self.n_features['BldPitch1']]])

        # Join libraries
        self.full_lib = {key: GeneralizedLibrary(all_libs[key],
                                            inputs_per_library=inputs_per_library[key])
                    for key in self.model_names}

def generate_turbulence_funcs(turb_class='A'):

    # Kaimal model, NTM, class B

    # turbulence intensity
    Iref = {'A': 0.16, 'B': 0.14, 'C': 0.12}
    b = 5.6

    # hub-height
    h = 60

    # standard deviation
    sigma = [0, 0, 0]
    sigma[0] = Iref[turb_class] * (0.75 * h + b)
    sigma[1] = 0.8 * sigma[0]
    sigma[2] = 0.5 * sigma[0]

    A1 = lambda z: 0.7 * z if z <= 60 else 42

    # asymptotically aooriach as frequency of intertial subrange increases
    # S1 = lambda f, z: 0.05 * sigma[0]**2 * (A1(z) / h)**(-2/3) * f**(-5/3)
    # S2 = lambda f, z: (4/ 3) * S1(f, z)
    # S3 = lambda f, z: (4/ 3) * S1(f, z)

    # velocity component integral scale parameter
    L = [0, 0, 0]
    L[0] = 8.1 * A1
    L[1] = 2.7 * A1
    L[2] = 8.66 * A1

    # power spectrical density
    S = [0, 0, 0]
    for k in range(3):
        S[k] = lambda f: (sigma[k]**2 / f) * (4 * f * L[k] / h) / (1 + 6 * f * L[k] / h)**(5 / 3)

    # TODO based on this model, simulate turbulent time-series at blade mid-point (z = (R/2)cos(Azimuth)) and
    #  use this as func of Azimuth and hub-height wind speed ? Manuel


def generate_firstord_lpf_var(rotspeed_ts, fc=0.125, dt=0.0125):
    ## generate a new filtered RotSpeed variable using the DT LPF

    omega_c = 2 * np.pi * fc
    num = [1 - np.exp(-dt * omega_c), 0]
    den = [1, -np.exp(-dt * omega_c)]
    lpf_op = lfilter(num, den, rotspeed_ts)
    return lpf_op

def generate_secord_lpf_var(windspeed_ts, omega=2*np.pi*0.025, zeta=0.707, dt=0.0125):
    num = [0, 0, omega**2]
    den = [1, 2 * omega * zeta, omega**2]
    _, lpf_op = lti(num, den).to_discrete(dt).output(windspeed_ts, [k * dt for k in range(len(windspeed_ts))], windspeed_ts[0])
    return lpf_op



def generate_drvts(key, case_idx, models, simulation_cases, state_cols, dt, filter=False):
    print(f'\nComputing BldPitch1 derivatives case {case_idx}')
    t_full, x_sim_full, _, _, _ = simulation_cases[key][case_idx]
    n_drvts = len(state_cols) - 1

    # Filter BldPitch and plot for single case
    order = 2
    cutoff = 0.5

    model = {}
    if filter:
        model[key] = butter_lowpass_filter(x_sim_full[:, state_cols['BldPitch1'].index('BldPitch1')],
                                                             cutoff, dt, order)
    else:
        model[key] = x_sim_full[:, state_cols['BldPitch1'].index('BldPitch1')]

    for d in range(n_drvts):
        model[f'{key}_{"d" * d}dot'] = models[key]._differentiate(model[f'{key}_{"d" * (d - 1)}dot'] if d > 0 else model[key])

    # BldPitch1_model['BldPitch1_dot'] = (BldPitch1_model['BldPitch1'][2:] - BldPitch1_model['BldPitch1'][0:-2]) \
    #                                    / (2 * dt)
    # BldPitch1_model['BldPitch1_ddot'] = (BldPitch1_model['BldPitch1'][2:] - 2 * BldPitch1_model['BldPitch1'][1:-1] +
    #                                      BldPitch1_model['BldPitch1'][0:-2]) / (2 * dt)
    # BldPitch1_model['BldPitch1'] = BldPitch1_model['BldPitch1'][1:-1]

    model['time'] = t_full[1:-1]

    return model

