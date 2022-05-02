import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from pysindy.feature_library import CustomLibrary, PolynomialLibrary, GeneralizedLibrary, FourierLibrary
import os
import matplotlib.pyplot as plt
from numpy.fft import fft
from sklearn.linear_model._base import safe_sparse_dot
FIGSIZE = (10, 8)

# def plot_fft(y, dt, op_label):
#     Y = fft(y)
#     N = len(Y)
#     n = range(N)
#     T = N * dt
#     freq = n / T
#
#     P2 = np.abs(Y / N)
#     P1 = P2[:(N / 2)]
#     P1[1:-1] = 2 * P1[1:-1]
#
#
#     fig, ax = plt.subplots(figsize=FIGSIZE)
#     ax.stem(freq, np.abs(Y), '#1f77b4', markerfmt=" ", basefmt="-b")
#     ax.set(xlabel="Frequency (Hz)", ylabel=f"FFT Amplitude of {op_label}")
#
#     return fig, ax

def plot_bldpitch1_mse(n_datasets, u_train, u_test, ctrl_inpt_cols, BldPitch1_model_filt, BldPitch1_model_unfilt):
    BldPitch1_drvts_mse_fig, BldPitch1_drvts_mse_axs = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True)
    BldPitch1_drvts_mse = {'BldPitch1_unfilt': [], 'BldPitch1_dot_unfilt': [], 'BldPitch1_ddot_unfilt': [],
                           'BldPitch1_filt': [], 'BldPitch1_dot_filt': [], 'BldPitch1_ddot_filt': []
                           }

    for case_idx in range(n_datasets):
        BldPitch1_drvts_mse['BldPitch1_unfilt'].append(compute_mse(
            data_true=np.concatenate([u_train['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1')],
                                      u_test['P_motor'][case_idx][:,
                                      ctrl_inpt_cols['P_motor'].index('BldPitch1')]])[1:-1],
            data_modeled=BldPitch1_model_unfilt[case_idx]['BldPitch1']))

        BldPitch1_drvts_mse['BldPitch1_filt'].append(compute_mse(
            data_true=np.concatenate(
                [u_train['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1')],
                 u_test['P_motor'][case_idx][:,
                 ctrl_inpt_cols['P_motor'].index('BldPitch1')]])[1:-1],
            data_modeled=BldPitch1_model_filt[case_idx]['BldPitch1']))

        BldPitch1_drvts_mse['BldPitch1_dot_unfilt'].append(compute_mse(
            data_true=np.concatenate(
                [u_train['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')],
                 u_test['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')]])[1:-1],
            data_modeled=BldPitch1_model_unfilt[case_idx]['BldPitch1_dot']))
        \
        BldPitch1_drvts_mse['BldPitch1_dot_filt'].append(compute_mse(
            data_true=np.concatenate(
                [u_train['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')],
                 u_test['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')]])[1:-1],
            data_modeled=BldPitch1_model_filt[case_idx]['BldPitch1_dot']))

        BldPitch1_drvts_mse['BldPitch1_ddot_unfilt'].append(compute_mse(
            data_true=np.concatenate(
                [u_train['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_ddot')],
                 u_test['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_ddot')]])[1:-1],
            data_modeled=BldPitch1_model_unfilt[case_idx]['BldPitch1_ddot']))

        BldPitch1_drvts_mse['BldPitch1_ddot_filt'].append(compute_mse(
            data_true=np.concatenate(
                [u_train['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_ddot')],
                 u_test['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_ddot')]])[1:-1],
            data_modeled=BldPitch1_model_filt[case_idx]['BldPitch1_ddot']))



    width = 0.25
    ind = np.array(range(1, n_datasets + 1))
    BldPitch1_drvts_mse_axs[0].bar(ind, BldPitch1_drvts_mse['BldPitch1_unfilt'], width,
                                   label='Unfiltered Modeled BldPitch1')
    BldPitch1_drvts_mse_axs[0].bar(ind + width, BldPitch1_drvts_mse['BldPitch1_filt'], width,
                                   label='Filtered Modeled BldPitch1')
    BldPitch1_drvts_mse_axs[0].set(ylabel='BldPitch1 MSE')

    BldPitch1_drvts_mse_axs[1].bar(ind, BldPitch1_drvts_mse['BldPitch1_dot_unfilt'], width,
                                   label='Unfiltered Modeled BldPitch1')
    BldPitch1_drvts_mse_axs[1].bar(ind + width, BldPitch1_drvts_mse['BldPitch1_dot_filt'], width,
                                   label='Filtered Modeled BldPitch1')
    BldPitch1_drvts_mse_axs[1].set(ylabel='BldPitch1_dot MSE')

    BldPitch1_drvts_mse_axs[2].bar(ind, BldPitch1_drvts_mse['BldPitch1_ddot_unfilt'], width,
                                   label='Unfiltered Modeled BldPitch1')
    BldPitch1_drvts_mse_axs[2].bar(ind + width, BldPitch1_drvts_mse['BldPitch1_ddot_filt'], width,
                                   label='Filtered Modeled BldPitch1')
    BldPitch1_drvts_mse_axs[2].set(ylabel='BldPitch1_ddot MSE')

    BldPitch1_drvts_mse_axs[-1].set(xlabel='Case Number')
    BldPitch1_drvts_mse_axs[-1].legend()

    return BldPitch1_drvts_mse_fig, BldPitch1_drvts_mse_axs, pd.DataFrame(BldPitch1_drvts_mse)


def generate_bldpitch1_true_drvts(case_idx, u_train, u_test, t_train, t_test, ctrl_inpt_cols):
    t_full = np.concatenate([t_train, t_test])
    return {
        'time': t_full,
        'BldPitch1': np.concatenate(
            [u_train['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1')],
             u_test['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1')]]),
        'BldPitch1_dot': np.concatenate(
            [u_train['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')],
             u_test['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')]]),
        'BldPitch1_ddot': np.concatenate(
            [u_train['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_ddot')],
             u_test['P_motor'][case_idx][:, ctrl_inpt_cols['P_motor'].index('BldPitch1_ddot')]])
    }


def plot_bldpitch_drvts_ts(test_idx, BldPitch1_model_unfilt, BldPitch1_model_filt, BldPitch1_true):
    BldPitch1_drvts_ts_fig, BldPitch1_drvts_ts_axs = plt.subplots(3, 2, figsize=FIGSIZE, sharex=True)

    modeled_start_idx = 100
    
    time_model = BldPitch1_model_unfilt[test_idx]['time']
    time_true = BldPitch1_true[test_idx]['time']

    BldPitch1_drvts_ts_axs[0][0].plot(time_model[modeled_start_idx:],
                                      BldPitch1_model_unfilt[test_idx]['BldPitch1'][modeled_start_idx:],
                                      linestyle="dashed", color='#1f77b4',
                                      label="Unfiltered Modeled BldPitch1")
    BldPitch1_drvts_ts_axs[0][1].plot(time_model[modeled_start_idx:],
                                      BldPitch1_model_filt[test_idx]['BldPitch1'][modeled_start_idx:],
                                      linestyle="dashed", color='#1f77b4',
                                      label="Filtered Modeled BldPitch1")
    BldPitch1_drvts_ts_axs[0][0].plot(time_true, BldPitch1_true[test_idx]['BldPitch1'],
                                      linestyle="solid", color='#ff7f0e',
                                      label="True BldPitch1")
    BldPitch1_drvts_ts_axs[0][1].plot(time_true, BldPitch1_true[test_idx]['BldPitch1'],
                                      linestyle="solid", color='#ff7f0e',
                                      label="True BldPitch1")
    BldPitch1_drvts_ts_axs[0][0].set(ylabel='BldPitch1')

    BldPitch1_drvts_ts_axs[1][0].plot(time_model[modeled_start_idx:],
                                      BldPitch1_model_unfilt[test_idx]['BldPitch1_dot'][modeled_start_idx:],
                                      linestyle="dashed", color='#1f77b4',
                                      label="Unfiltered Modeled BldPitch1")
    BldPitch1_drvts_ts_axs[1][1].plot(time_model[modeled_start_idx:],
                                      BldPitch1_model_filt[test_idx]['BldPitch1_dot'][modeled_start_idx:],
                                      linestyle="dashed", color='#1f77b4',
                                      label="Filtered Modeled BldPitch1")
    BldPitch1_drvts_ts_axs[1][0].plot(time_true, BldPitch1_true[test_idx]['BldPitch1_dot'],
                                      linestyle="solid", color='#ff7f0e',
                                      label="True BldPitch1")
    BldPitch1_drvts_ts_axs[1][1].plot(time_true, BldPitch1_true[test_idx]['BldPitch1_dot'],
                                      linestyle="solid", color='#ff7f0e',
                                      label="True BldPitch1")
    BldPitch1_drvts_ts_axs[1][0].set(ylabel='BldPitch1_dot')

    BldPitch1_drvts_ts_axs[2][0].plot(time_model[modeled_start_idx:],
                                      BldPitch1_model_unfilt[test_idx]['BldPitch1_ddot'][modeled_start_idx:],
                                      linestyle="dashed", color='#1f77b4',
                                      label="Unfiltered Modeled BldPitch1")
    BldPitch1_drvts_ts_axs[2][1].plot(time_model[modeled_start_idx:],
                                      BldPitch1_model_filt[test_idx]['BldPitch1_ddot'][modeled_start_idx:],
                                      linestyle="dashed", color='#1f77b4',
                                      label="Filtered Modeled BldPitch1")
    BldPitch1_drvts_ts_axs[2][0].plot(time_true,
                                      BldPitch1_true[test_idx]['BldPitch1_ddot'],
                                      linestyle="solid", color='#ff7f0e',
                                      label="True BldPitch1")
    BldPitch1_drvts_ts_axs[2][1].plot(time_true,
                                      BldPitch1_true[test_idx]['BldPitch1_ddot'],
                                      linestyle="solid", color='#ff7f0e',
                                      label="True BldPitch1")

    BldPitch1_drvts_ts_axs[2][0].set(ylabel='BldPitch1_ddot')

    BldPitch1_drvts_ts_axs[0][0].legend()
    BldPitch1_drvts_ts_axs[0][1].legend()

    BldPitch1_drvts_ts_axs[-1][0].set(xlabel='$k$')
    return BldPitch1_drvts_ts_fig, BldPitch1_drvts_ts_axs

def compute_mse(data_true, data_modeled):
    n_datasets = len(data_true)
    return (1 / n_datasets) * np.sum([(y_true - y_mod)**2
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

def generate_candidate_funcs(model_names, n_features, feature_cols):
    all_libs = {key: [] for key in model_names}
    inputs_per_library = {key: np.zeros((0, n_features[key]), dtype=int) for key in model_names}

    # Polynomial of all but GenSpeed
    # poly_lib1 = {key: PolynomialLibrary(degree=2, include_interaction=True, include_bias=True) for key in model_names}
    # all_libs['P_motor'].append(poly_lib1['P_motor'])
    # idx = np.array([1] + list(range(1, n_features['P_motor'])))
    # idx[feature_cols['P_motor'].index('GenSpeed')] = feature_cols['P_motor'].index('GenSpeed') + 1
    # idx[feature_cols['P_motor'].index('GenSpeed_dev_rated')] = idx[feature_cols['P_motor'].index('GenSpeed')]
    # idx[feature_cols['P_motor'].index('Wind1VelX_dev_rated')] = idx[feature_cols['P_motor'].index('Wind1VelX')]
    # idx[feature_cols['P_motor'].index('Wind1VelX_dev')] = idx[feature_cols['P_motor'].index('Wind1VelX')]
    # inputs_per_library['P_motor'] = np.vstack([inputs_per_library['P_motor'], idx])
    #
    # all_libs['BldPitch1'].append(poly_lib1['BldPitch1'])
    # idx = np.array([1] + list(range(1, n_features['BldPitch1'])))
    # idx[feature_cols['BldPitch1'].index('GenSpeed')] = feature_cols['BldPitch1'].index('GenSpeed') + 1
    # idx[feature_cols['BldPitch1'].index('GenSpeed_dev_rated')] = idx[feature_cols['BldPitch1'].index('GenSpeed')]
    # idx[feature_cols['BldPitch1'].index('Wind1VelX_dev_rated')] = idx[feature_cols['BldPitch1'].index('Wind1VelX')]
    # idx[feature_cols['BldPitch1'].index('Wind1VelX_dev')] = idx[feature_cols['BldPitch1'].index('Wind1VelX')]
    # inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'], idx])

    # Polynomial of all but GenTq
    poly_lib2 = {key: PolynomialLibrary(degree=2, include_interaction=True, include_bias=False) for key in model_names}
    all_libs['P_motor'].append(poly_lib2['P_motor'])
    idx = np.array([1] + list(range(1, n_features['P_motor'])))
    idx[feature_cols['P_motor'].index('GenTq')] = feature_cols['P_motor'].index('GenTq') + 1
    idx[feature_cols['P_motor'].index('GenSpeed_dev_rated')] = idx[feature_cols['P_motor'].index('GenSpeed')]
    idx[feature_cols['P_motor'].index('Wind1VelX_dev_rated')] = idx[feature_cols['P_motor'].index('Wind1VelX')]
    idx[feature_cols['P_motor'].index('Wind1VelX_dev')] = idx[feature_cols['P_motor'].index('Wind1VelX')]
    inputs_per_library['P_motor'] = np.vstack([inputs_per_library['P_motor'], idx])

    all_libs['BldPitch1'].append(poly_lib2['BldPitch1'])
    idx = np.array([1] + list(range(1, n_features['BldPitch1'])))
    idx[feature_cols['BldPitch1'].index('GenTq')] = feature_cols['BldPitch1'].index('GenTq') + 1
    idx[feature_cols['BldPitch1'].index('GenSpeed_dev_rated')] = idx[feature_cols['BldPitch1'].index('GenSpeed')]
    idx[feature_cols['BldPitch1'].index('Wind1VelX_dev_rated')] = idx[feature_cols['BldPitch1'].index('Wind1VelX')]
    idx[feature_cols['BldPitch1'].index('Wind1VelX_dev')] = idx[feature_cols['BldPitch1'].index('Wind1VelX')]
    inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'], idx])

    # RotSpeed terms
    # genspeed_idx = {key: [feature_cols[key].index('GenSpeed')] for key in model_names}
    # genspeed_poly_lib = {key: PolynomialLibrary(degree=2, include_interaction=True, include_bias=True) for key in
    #                      model_names}
    # all_libs['P_motor'].append(genspeed_poly_lib['P_motor'])
    # inputs_per_library['P_motor'] = np.vstack(
    #     [inputs_per_library['P_motor'], (genspeed_idx['P_motor'] * n_features['P_motor'])[:n_features['P_motor']]])
    #
    # all_libs['BldPitch1'].append(genspeed_poly_lib['BldPitch1'])
    # inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'],
    #                                              (genspeed_idx['BldPitch1'] * n_features['BldPitch1'])[
    #                                              :n_features['BldPitch1']]])

    # TipDefl terms
    # tipdefl_idx = {key: [feature_cols[key].index('OoPDefl1'),
    #                      feature_cols[key].index('IPDefl1')] for key in model_names}
    # tipdefl_poly_lib = {key: PolynomialLibrary(degree=2, include_interaction=False, include_bias=False) for key in
    #                     ['P_motor']}
    # all_libs['P_motor'].append(tipdefl_poly_lib['P_motor'])
    # inputs_per_library['P_motor'] = np.vstack([inputs_per_library['P_motor'],
    #                                            (tipdefl_idx['P_motor'] * n_features['P_motor'])[
    #                                            :n_features['P_motor']]])

    # BlPitch1, BlPitch1_dot, BlPitch1_ddot terms
    # beta_drvts_idx = {key: [feature_cols[key].index('BldPitch1'),
    #                         feature_cols[key].index('BldPitch1_dot'),
    #                         feature_cols[key].index('BldPitch1_ddot')] for key in ['P_motor']}
    # beta_drvts_poly_lib = {key: PolynomialLibrary(degree=2, include_interaction=True, include_bias=False) for key in
    #                        ['P_motor']}
    # all_libs['P_motor'].append(beta_drvts_poly_lib['P_motor'])
    # inputs_per_library['P_motor'] = np.vstack([inputs_per_library['P_motor'],
    #                                            (beta_drvts_idx['P_motor'] * n_features['P_motor'])[
    #                                            :n_features['P_motor']]])

    # Sinusoidal theta terms - \sin(\theta), \cos(\theta), \sin(\theta)\cos(\theta), \sin^2(\theta)
    # theta_idx = {key: [feature_cols[key].index('Azimuth')] for key in model_names}
    sin_lib_funcs = [
        lambda x: np.sin(x) * np.cos(x),
        lambda x: np.sin(x)**2,
        lambda x: np.sin(x),
        lambda x: np.cos(x)
    ]
    sin_lib_func_names = [lambda x: f"sin({x})cos({x})",
                            lambda x: f"sin({x})^2",
                            lambda x: f"sin({x})",
                            lambda x: f"cos({x})"
                            ]

    theta_lib = {key: CustomLibrary(library_functions=sin_lib_funcs, function_names=sin_lib_func_names) for key in
                 model_names}
    all_libs['P_motor'].append(theta_lib['P_motor'])
    inputs_per_library['P_motor'] = np.vstack([inputs_per_library['P_motor'],
                                               [1] + list(range(1, n_features['P_motor']))])

    all_libs['BldPitch1'].append(theta_lib['BldPitch1'])
    inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'],
                                                 [1] + list(range(1, n_features['BldPitch1']))])

    # Beta_bar schedule terms: (max(0, V - Vrated))^1/2, (max(0, V - Vrated))^1/3, max(0, V - Vrated)
    v_dev_rated_idx = {key: [feature_cols[key].index('Wind1VelX_dev_rated')] for key in ['BldPitch1']}

    # TODO scatter BldPitch1 vs highest ranking terms
    beta0_lib_funcs = [
        lambda x: (np.maximum(np.zeros_like(x), x)) ** (1 / 2),
        lambda x: (np.maximum(np.zeros_like(x), x)) ** (1 / 3),
        lambda x: np.maximum(np.zeros_like(x), x)]

    beta0_lib_func_names = [lambda x: f"max(0, {x})^(1/2)",
                            lambda x: f"max(0, {x})^(1/3)",
                            lambda x: f"max(0, {x})"]
    beta0_lib = {key: CustomLibrary(library_functions=beta0_lib_funcs, function_names=beta0_lib_func_names) for key in
                 ['BldPitch1']}
    all_libs['BldPitch1'].append(beta0_lib['BldPitch1'])
    inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'],
                                                 (v_dev_rated_idx['BldPitch1'] * n_features['BldPitch1'])[
                                                 :n_features['BldPitch1']]])

    # Wind speed disturbance V - Vbar term
    # v_dev_idx = {key: [feature_cols[key].index('Wind1VelX_dev')] for key in ['BldPitch1']}
    # v_dev_lib = {key: PolynomialLibrary(degree=2, include_interaction=False, include_bias=False) for key in
    #              ['BldPitch1']}
    # all_libs['BldPitch1'].append(v_dev_lib['BldPitch1'])
    # inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'],
    #                                              (v_dev_idx['BldPitch1'] * n_features['BldPitch1'])[
    #                                              :n_features['BldPitch1']]])

    # Generator speed deviation omega_g - omega_g_bar term
    # omega_dev_rated_idx = {key: [feature_cols[key].index('GenSpeed_dev_rated')] for key in ['BldPitch1']}
    # omega_delta_lib = {key: PolynomialLibrary(degree=2, include_interaction=False, include_bias=False) for key in
    #                    ['BldPitch1']}
    # all_libs['BldPitch1'].append(omega_delta_lib['BldPitch1'])
    # inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'],
    #                                              (omega_dev_rated_idx['BldPitch1'] * n_features['BldPitch1'])[
    #                                              :n_features['BldPitch1']]])

    # Fourier library of all variables
    # fourier_lib = {key: FourierLibrary(n_frequencies=2, include_cos=True, include_sin=True) for key in ['BldPitch1']}
    # all_libs['BldPitch1'].append(fourier_lib['BldPitch1'])
    # inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'], np.arange(n_features['BldPitch1'])])

    # TODO check if close to equilibrium or oscillating

    # Define inputs for each library
    n_libraries = {key: len(libs) for key, libs in all_libs.items()}

    # Join libraries
    full_lib = {key: GeneralizedLibrary(all_libs[key],
                                        inputs_per_library=inputs_per_library[key])
                for key in model_names}

    return full_lib

def process_data(data_all, n_datasets, model_names):
    ## Export Datasets to Arrays
    # V_mean = [12, 14, 20]
    # dataset_indices = [i for i in range(len(data_all['Wind1VelX_mean'])) if data_all['Wind1VelX_mean'][i] in V_mean]

    dataset_indices = range(n_datasets)
    # subset_in_channels = ['tt', 'RotSpeed', 'Wind1VelX', 'BldPitch1', 'BldPitch1_dot', 'BldPitch1_ddot', 'Azimuth',
    #                       'RootMzc1', 'OoPDefl1', 'IPDefl1', 'GenSpeed', 'P_motor', 'Wind1VelX_dev', 'Wind1VelX_dev_rated', 'GenSpeed_dev_rated']
    all_channels = list(data_all['training_input_data'][0].keys())

    exclude_channels = ['GenPwr', 'RotSpeed',
                        'RotTorq']  # 'RtTSR', 'RotThrust' # ? maybe remove V-Vr, GenSpeed - GenSpeedr if can be easily shifted by constatn
    subset_in_channels = np.array(
        [x for x in all_channels if '2' not in x and '3' not in x and x not in exclude_channels])

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
        'BldPitch1',
        'GenTq',
        # 'RtAeroCp',
        # 'RotThrust',
        # 'RotSpeed',
        'GenSpeed',
        'OoPDefl1', 'IPDefl1',
        # 'TwrBsM', #'TwrBsMyt', 'TwrBsMxt',
        # 'TwrClrnc1',
        # 'TTDspFA', 'TTDspSS',
        # 'YawBrM', #'YawBrMzp', 'YawBrMyp', 'YawBrMxp',
        # 'Wind1VelY', 'Wind1VelZ',
        'Azimuth',
        'BldPitch1_dot', 'BldPitch1_ddot',
        'P_motor',
        'Wind1VelX_dev', 'Wind1VelX_dev_rated', 'GenSpeed_dev_rated'
    ]

    # subset_in_channels = np.delete(all_channels, exclude_channel_idx)
    dataset_train_df = [pd.DataFrame(data_all['training_input_data'][i], columns=subset_in_channels)
                        for i in dataset_indices]
    dataset_test_df = [pd.DataFrame(data_all['testing_input_data'][i], columns=subset_in_channels)
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

    ctrl_inpt_cols = {'P_motor': [c for c in subset_in_channels if c not in ['tt', 'P_motor']],
                      'BldPitch1': [c for c in subset_in_channels if
                                    c not in ['tt', 'BldPitch1', 'P_motor', 'BldPitch1_dot', 'BldPitch1_ddot']]}

    state_cols = {
        'P_motor': [c for c in subset_in_channels if c not in ctrl_inpt_cols['P_motor'] + ['tt']],
        'BldPitch1': [c for c in subset_in_channels
                      if c not in ctrl_inpt_cols['BldPitch1'] + ['BldPitch1_dot', 'BldPitch1_ddot', 'P_motor'] + [
                          'tt']]}

    t_train = dataset_train_df[0]['tt'].to_numpy()
    t_test = dataset_test_df[0]['tt'].to_numpy()

    u_train = {key: [dataset_train_df[i].loc[:, ctrl_inpt_cols[key]].to_numpy()
                     for i in range(len(dataset_indices))]
               for key in model_names}
    u_test = {key: [dataset_test_df[i].loc[:, ctrl_inpt_cols[key]].to_numpy()
                    for i in range(len(dataset_indices))]
              for key in model_names}

    x_train = {key: [dataset_train_df[i].loc[:, state_cols[key]].to_numpy()
                     for i in range(len(dataset_indices))]
               for key in model_names}
    x_test = {key: [dataset_test_df[i].loc[:, state_cols[key]].to_numpy()
                    for i in range(len(dataset_indices))]
              for key in model_names}

    feature_cols = {key: state_cols[key] + ctrl_inpt_cols[key] for key in model_names}
    n_features = {key: len(feature_cols[key]) for key in model_names}

    return t_train, t_test, u_train, u_test, x_train, x_test, feature_cols, state_cols, ctrl_inpt_cols, n_features

def load_data(data_dir, vmean_list, seed_list, V_rated, GenSpeed_rated):
    n_wind = len(vmean_list)
    n_seed = len(seed_list)
    n_datasets = n_wind * n_seed

    data_all = {'P_motor_mean_seed': [], 'P_motor_mean_vmean': [],
                'P_motor_max_seed': [], 'P_motor_max_vmean': [],
                'seed': [], 'vmean': [], 'filename': [], 'dt': [],
                'training_input_data': [], 'testing_input_data': [],
                'training_output_data': [], 'testing_output_data': [],
                'n_datapoints': []}

    # for each mean wind speed
    for v_idx in range(len(vmean_list)):
        # for each turbulence seed
        for s_idx in range(len(seed_list)):

            # load the data
            data_filename = os.path.join(data_dir, f'B_{vmean_list[v_idx]}_{seed_list[s_idx]}.mat')
            print(f'> loading {data_filename} ..\n')
            data = loadmat(data_filename)

            # extract the output channel names
            if v_idx == 0 and s_idx == 0:
                op_channels = [ch[0] for ch in data['Chan'][0].dtype.descr]

            # create a dictionary of the data
            data_dict = {ch_name: data['Chan'][0][0][ch_idx] for ch_idx, ch_name in enumerate(op_channels)}

            # generate beta_dot and beta_ddot values
            init_idx = 5000
            dt = (data_dict['tt'][1] - data_dict['tt'][0])[0]

            beta_dot = [
                (data_dict[f'BldPitch{b + 1}'][init_idx + 1:] - data_dict[f'BldPitch{b + 1}'][init_idx - 1:-2]) \
                / (2 * dt) for b in range(3)]
            beta_ddot = [
                (data_dict[f'BldPitch{b + 1}'][init_idx + 1:] - 2 * data_dict[f'BldPitch{b + 1}'][init_idx:-1]
                 + data_dict[f'BldPitch{b + 1}'][init_idx - 1:-2]) / (2 * dt) for b in range(3)]

            # pre-truncate data
            for k, v in data_dict.items():
                data_dict[k] = v[init_idx:-1]

            # for b in range(3):
            #     data_dict[f'BldPitch{b+1}_dot'] = beta_dot[b]
            #     data_dict[f'BldPitch{b+1}_ddot'] = beta_ddot[b]
            data_dict[f'BldPitch1_dot'] = beta_dot[0]
            data_dict[f'BldPitch1_ddot'] = beta_ddot[0]

            P_motor = data_dict['RootMzc1'] * data_dict['BldPitch1_dot']
            data_dict['P_motor'] = P_motor

            data_dict['Wind1VelX_dev'] = data_dict['Wind1VelX'] - vmean_list[v_idx]
            data_dict['Wind1VelX_dev_rated'] = data_dict['Wind1VelX'] - V_rated
            data_dict['GenSpeed_dev_rated'] = data_dict['GenSpeed'] - GenSpeed_rated

            # squeeze data
            for k, v in data_dict.items():
                data_dict[k] = np.squeeze(v)

            n_datapoints = len(data_dict['tt'])

            # compute true bladepitch motor power

            P_motor_mean_seed = np.mean(P_motor)
            P_motor_max_seed = (np.max(np.abs(P_motor)) * (P_motor[np.argmax(np.abs(P_motor))] / np.abs(P_motor[np.argmax(np.abs(P_motor))])))[0]
            if np.max(np.abs(P_motor)) == 0:
                P_motor_max_seed = 0.0
            data_dict_output = {}
            data_dict_output['P_motor'] = P_motor
            data_dict_output['BldPitch1'] = data_dict['BldPitch1']

            # add parameters of this dataset
            # data_all['Wind1VelX_mean'].append(vmean_list[v_idx])
            # data_all['GenSpeed_mean'].append(np.mean(data_dict['GenSpeed']))
            data_all['seed'].append(seed_list[s_idx])
            data_all['vmean'].append(vmean_list[v_idx])
            data_all['dt'].append(dt)
            # data_all['Wind1VelX_std'].append(np.std(data_dict['Wind1VelX']))
            data_all['filename'].append(data_filename)
            data_all['P_motor_mean_seed'].append(P_motor_mean_seed)
            data_all['P_motor_max_seed'].append(P_motor_max_seed)
            data_all['n_datapoints'].append(n_datapoints)

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
            data_all['training_input_data'].append(data_dict_training)
            data_all['testing_input_data'].append(data_dict_testing)

            # add output datasets
            data_dict_training_output = {}
            data_dict_testing_output = {}
            for k, v in data_dict_output.items():
                data_dict_training_output[k] = v[training_idx]
                data_dict_testing_output[k] = v[testing_idx]
            data_all['training_output_data'].append(data_dict_training_output)
            data_all['testing_output_data'].append(data_dict_testing_output)

        P_motor_mean_vmean = np.mean(
            [P_val for vmean, P_val in zip(data_all['vmean'], data_all['P_motor_mean_seed'])
             if vmean == vmean_list[v_idx]])
        P_motor_max_vmean = np.mean(
            [P_val for vmean, P_val in zip(data_all['vmean'], data_all['P_motor_max_seed'])
             if vmean == vmean_list[v_idx]])
        data_all['P_motor_mean_vmean'] = data_all['P_motor_mean_vmean'] + ([P_motor_mean_vmean] * n_seed)
        data_all['P_motor_max_vmean'] = data_all['P_motor_max_vmean'] + ([P_motor_max_vmean] * n_seed)

    return data_all


def plot_ensemble_error(models, key, m_idx, state_cols, op_idx, ensemble_axs):
    median_ensemble = np.median(models[key].coef_list, axis=0)

    # mean_ensemble = np.mean(models[key].coef_list, axis=0)
    std_ensemble = np.std(models[key].coef_list, axis=0)
    mean_ensemble = np.mean(models[key].coef_list, axis=0)
    candidate_funcs = models[key].get_feature_names()

    xticknames = [f'${f}$' for f in models[key].get_feature_names()]

    sorted_coeff_indices = np.flip(np.argsort(np.abs(std_ensemble[op_idx, :])))
    mean_ensemble = np.array(mean_ensemble[op_idx, :])[sorted_coeff_indices]
    std_ensemble = np.array(std_ensemble[op_idx, :])[sorted_coeff_indices]
    candidate_funcs = np.array(candidate_funcs)[sorted_coeff_indices]
    coeffs_df = pd.DataFrame(
        data={'Candidate Function': candidate_funcs, 'Mean': mean_ensemble, 'Standard Deviation': std_ensemble})
    print(f"\n{coeffs_df}")

    # for i in range(len(state_cols[key])):
    ensemble_axs[m_idx].errorbar(range(1, len(models[key].get_feature_names()) + 1), median_ensemble[op_idx, :],
                                 yerr=std_ensemble,
                                 fmt='o')
    ensemble_axs[m_idx].set(title=f'Ensembling for {state_cols[key][op_idx]}')
    return coeffs_df



def plot_correlations(key, case_idx, m_idx, op_idx, traj_axs, traj_labels, ctrl_inpt_cols, x_sim_train, x_sim_test,
                      x_train, x_test, u_train, u_test):
    scatter_size = 0.1
    op_label = key
    for ax_idx, inp_label in enumerate(traj_labels[key]):
        inp_idx = ctrl_inpt_cols[key].index(inp_label)
        # Plot trajectories over training phase
        traj_axs[ax_idx + m_idx, 0].scatter(u_train[key][case_idx][:, inp_idx], x_sim_train[:, op_idx],
                                            linestyle="dashed", s=scatter_size,
                                            label="Model Simulation")
        traj_axs[ax_idx + m_idx, 0].scatter(u_train[key][case_idx][:, inp_idx], x_train[key][case_idx][:, op_idx],
                                            linestyle="solid", s=scatter_size,
                                            label="True Simulation")
        traj_axs[ax_idx + m_idx, 0].set(xlabel=f"{inp_label}", ylabel=f"{op_label}", title="Training Phase")
        traj_axs[ax_idx + m_idx, 0].legend()

        # Plot trajectories over testing phase
        traj_axs[ax_idx + m_idx, 1].scatter(u_test[key][case_idx][:, inp_idx], x_sim_test[:, op_idx],
                                            linestyle="dashed", s=scatter_size,
                                            label="Model Simulation")
        traj_axs[ax_idx + m_idx, 1].scatter(u_test[key][case_idx][:, inp_idx], x_test[key][case_idx][:, op_idx],
                                            linestyle="solid", s=scatter_size,
                                            label="True Simulation")
        traj_axs[ax_idx + m_idx, 1].set(xlabel=f"{inp_label}", ylabel=f"{op_label}", title="Testing Phase")
        traj_axs[ax_idx + m_idx, 1].legend()



def plot_time_series(key, case_idx, m_idx, op_idx, t_train, t_sim, t_true, x_train, x_test, x_sim,
                     sim_ts_axs, full_model=False):
    op_label = key
    x_true = np.concatenate([x_train[key][case_idx][:, op_idx], x_test[key][case_idx][:, op_idx]])[:, np.newaxis]
    x_all = np.hstack([x_sim[:, op_idx].T, x_true[:, op_idx].T])
    x_min = np.min(x_all)
    x_max = np.max(x_all)

    if full_model:
        # remove outliers
        std = np.std(x_sim[:, op_idx])
        mean = np.mean(x_sim[:, op_idx])
        inlier_idx = np.where((x_sim[:, op_idx] < mean + (0.005 * std)) & (x_sim[:, op_idx] > mean - (0.005 * std)))


        start_idx = 100
        x_all = np.concatenate([x_sim[inlier_idx, op_idx].squeeze(), x_true[:, op_idx]])
        x_min = np.min(x_all)
        x_max = np.max(x_all)
        x_mean = np.mean(x_all)
        # sim_ts_fig, sim_ts_axs = plt.subplots(3, 1)
        sim_ts_axs[2].plot(t_sim[inlier_idx], x_sim[inlier_idx, op_idx].squeeze(), linestyle="dashed", label="Full Model Simulation")
        sim_ts_axs[2].plot(t_true, x_true, linestyle="solid", label="True Simulation")
        sim_ts_axs[2].plot([t_train[-1], t_train[-1]], [x_min, x_max], linestyle='dotted',
                           label="Train/Test Boundary")
        sim_ts_axs[2].set(xlabel='$k$', ylabel=f'$P_{{motor}}$')
        # sim_ts_fig.show()
    elif key == 'P_motor':
        sim_ts_axs[m_idx].plot(t_sim, x_sim, linestyle="dashed", label="Model Simulation")
        sim_ts_axs[m_idx].plot(t_sim, x_true, linestyle="solid", label="True Simulation")
        sim_ts_axs[m_idx].plot([t_train[-1], t_train[-1]], [x_min, x_max], linestyle='dotted',
                               label="Train/Test Boundary")
        sim_ts_axs[m_idx].set(xlabel='$k$', ylabel=op_label)
    else:
        sim_ts_axs[m_idx].plot(t_sim, x_sim, linestyle="dashed", label="Model Simulation")
        sim_ts_axs[m_idx].plot(t_sim, x_true, linestyle="solid", label="True Simulation")
        sim_ts_axs[m_idx].plot([t_train[-1], t_train[-1]], [x_min, x_max], linestyle='dotted',
                               label="Train/Test Boundary")
        sim_ts_axs[m_idx].set(xlabel='$k$', ylabel=op_label)


    sim_ts_axs[0].legend()


def plot_coeffs(models, key, m_idx, op_idx, coeff_axs):
    # Plot barchart of coefficients for different terms for BlPitch1 and P_motor
    op_label = key
    candidate_funcs = models[key].get_feature_names()
    coeffs = models[key].coefficients()[op_idx, :]
    sorted_coeff_indices = np.flip(np.argsort(np.abs(coeffs)))
    candidate_funcs = np.array(candidate_funcs)[sorted_coeff_indices]
    coeffs = np.array(coeffs)[sorted_coeff_indices]

    coeff_axs[m_idx].bar(range(1, len(candidate_funcs) + 1), coeffs)
    coeff_axs[m_idx].set(xlabel='Feature Index', ylabel='Coefficient Value', title=op_label)

    coeffs_df = pd.DataFrame(
        data={'Candidate Function': candidate_funcs, 'Coefficient': coeffs})

    print(f"\n{coeffs_df}")
    # print(f'\n{op_label} Coefficients in order of absolute value:')
    # for ii in range(len(coeffs)):
    #     print(f'{candidate_funcs[ii]} - {coeffs[ii]}')
    return coeffs_df


def generate_BldPitch1_drvts(case_idx, BldPitch1_sim, state_cols, dt, filter=False):
    print(f'\nComputing BldPitch1 derivatives case {case_idx}')

    t_full, x_sim_full, _, _, _ = BldPitch1_sim[case_idx]

    # Filter BldPitch and plot for single case
    order = 2
    cutoff = 0.5

    BldPitch1_model = {}
    if filter:
        BldPitch1_model['BldPitch1'] = butter_lowpass_filter(x_sim_full[:, state_cols['BldPitch1'].index('BldPitch1')], cutoff, dt, order)
    else:
        BldPitch1_model['BldPitch1'] = x_sim_full[:, state_cols['BldPitch1'].index('BldPitch1')]

    BldPitch1_model['BldPitch1_dot'] = (BldPitch1_model['BldPitch1'][2:] - BldPitch1_model['BldPitch1'][0:-2]) \
                                       / (2 * dt)
    BldPitch1_model['BldPitch1_ddot'] = (BldPitch1_model['BldPitch1'][2:] - 2 * BldPitch1_model['BldPitch1'][1:-1] +
                                         BldPitch1_model['BldPitch1'][0:-2]) / (2 * dt)
    BldPitch1_model['BldPitch1'] = BldPitch1_model['BldPitch1'][1:-1]

    BldPitch1_model['time'] = t_full[1:-1]

    return BldPitch1_model


def simulate_case(key, test_idx, ctrl_inpt_cols,
                  t_train, t_test, x_train, x_test, u_train, u_test,
                  models, ensemble_coeffs, full_lib,
                  full_model=False, BldPitch1_model=None):
    print(f'\nSimulating {key} for case {test_idx}')

    # full_lib.library_ensemble = False

    x0_train = x_train[key][test_idx][0, :]
    x0_test = x_test[key][test_idx][0, :]
    t_full = np.concatenate([t_train, t_test])

    if key == 'P_motor' and full_model:
        x0_train = x_train[key][test_idx][1, :]

        u_train_dash = np.array(u_train[key][test_idx][1:])
        u_test_dash = np.array(u_test[key][test_idx][:-1])
        u_train_dash[:, ctrl_inpt_cols['P_motor'].index('BldPitch1')] = BldPitch1_model[test_idx]['BldPitch1'][
                                                                        :len(t_train) - 1]
        u_train_dash[:, ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')] = BldPitch1_model[test_idx]['BldPitch1_dot'][
                                                                            :len(t_train) - 1]
        u_train_dash[:, ctrl_inpt_cols['P_motor'].index('BldPitch1_ddot')] = BldPitch1_model[test_idx]['BldPitch1_ddot'][
                                                                             :len(t_train) - 1]
        u_test_dash[:, ctrl_inpt_cols['P_motor'].index('BldPitch1')] = BldPitch1_model[test_idx]['BldPitch1'][
                                                                       len(t_train) - 1:]
        u_test_dash[:, ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')] = BldPitch1_model[test_idx]['BldPitch1_dot'][
                                                                           len(t_train) - 1:]
        u_test_dash[:, ctrl_inpt_cols['P_motor'].index('BldPitch1_ddot')] = BldPitch1_model[test_idx]['BldPitch1_ddot'][
                                                                            len(t_train) - 1:]

        truncate_idx = 1
        x0_train = x_train[key][test_idx][1, :]
        t_train = np.array(t_train[1:])
        t_test = np.array(t_test[:-1])

        # models[key].simulate(x0_train, len(t_train) - 2, u=u_train_dash)
        # x_sim_test = models[key].simulate(x0_test, len(t_test) - 2, u=u_test_dash)


    else:
        u_train_dash = u_train[key][test_idx]
        u_test_dash = u_test[key][test_idx]

        truncate_idx = 0

        # x_sim_train2 = models[key].simulate(x0_train, len(t_train), u=u_train_dash)
        # x_sim_test2 = models[key].simulate(x0_test, len(t_test), u=u_test_dash)

    # for i in range(1, t):
    #     x[i] = models[key].predict(x[i - 1: i], u=u[i - 1, np.newaxis])

    # x_shape = np.shape(x)
    # x = validate_input(x)
    # u = validate_control_variables(x, u)
    # return self.model.predict(np.concatenate((x, u), axis=1)).reshape(
    #     x_shape
    # )

    # Xt = X = np.concatenate((xk, uk))
    # for _, name, transform in models[key].model._iter(with_final=False):
    #     Xt = transform.transform(Xt)
    # return models[key].model.steps[-1][1].predict(Xt, **{})

    # prediction = models[key].model.steps[-1][1].optimizer.predict(x)
    # if prediction.ndim == 1:
    #     return prediction[:, np.newaxis]
    # models[key].model.steps[-1][1].optimizer.predict
    # models[key].model.steps[-1][1].optimizer.coef_ - ensemble_coeffs[key] # yes

    x_sim_train = np.zeros((len(t_train), len(x0_train)))
    x_sim_train[0, :] = x0_train
    for k in range(1, len(t_train)):
        xk = x_sim_train[k - 1, :]
        uk = u_train_dash[k - 1, :]
        Xt = np.concatenate((xk, uk))[np.newaxis, :]
        for _, name, transform in models[key].model._iter(with_final=False):
            Xt = transform.transform(Xt)
        x_sim_train[k, :] = safe_sparse_dot(Xt,
                                            models[key].model.steps[-1][1].optimizer.coef_.T, dense_output=True) \
                            + models[key].model.steps[-1][1].optimizer.intercept_
            #models[key].model.steps[-1][1].predict(Xt, **{})
        # x_sim_train[k, :] = ensemble_coeffs[key] @ full_lib.transform(np.concatenate([xk, uk])[np.newaxis, :]).T

    x_sim_test = np.zeros((len(t_test), len(x0_test)))
    x_sim_test[0, :] = x0_test
    for k in range(1, len(t_test)):
        xk = x_sim_test[k - 1, :]
        uk = u_test_dash[k - 1, :]
        Xt = np.concatenate((xk, uk))[np.newaxis, :]
        for _, name, transform in models[key].model._iter(with_final=False):
            Xt = transform.transform(Xt)
        x_sim_test[k, :] = safe_sparse_dot(Xt,
                                            models[key].model.steps[-1][1].optimizer.coef_.T, dense_output=True) \
                            + models[key].model.steps[-1][1].optimizer.intercept_

    if truncate_idx:
        t_full = t_full[truncate_idx:-truncate_idx]

    # t = len(t_train)
    #
    # x_sim_train = np.zeros((t, models[key]._n_features_in_ - models[key].n_control_features_))
    # x_sim_train[0] = x0_train
    # for i in range(1, t):
    #     x_sim_train[i] = models[key].predict(x_sim_train[i - 1: i], u=u_train[key][case_idx][i - 1, np.newaxis])
    #
    # x_sim_test = np.zeros((t, models[key]._n_features_in_ - models[key].n_control_features_))
    # x_sim_test[0] = x0_test
    # for i in range(1, t):
    #     x_sim_test[i] = model_predict_func(x_sim_test[i - 1: i], u=u_test[key][case_idx][i - 1, np.newaxis])

    x_sim_full = np.concatenate([x_sim_train, x_sim_test])
    u_sim_full = np.concatenate([u_train_dash, u_test_dash])

    return t_full, x_sim_full, u_sim_full, x_sim_train, x_sim_test

