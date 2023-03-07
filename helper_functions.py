import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, lfilter, lti, dlti
from pysindy.feature_library import CustomLibrary, PolynomialLibrary, GeneralizedLibrary, FourierLibrary
import os
from pysindy import FiniteDifference
import pickle
import matplotlib.pyplot as plt
from pyFAST.input_output.turbsim_file import TurbSimFile
import numpy as np
from scipy.interpolate import interp1d

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


def generate_firstord_lpf_var(rotspeed_ts, fc=0.125, dt=0.0125, init_ip=2):
    ## generate a new filtered RotSpeed variable using the DT LPF

    omega_c = 2 * np.pi * fc
    num = [1 - np.exp(-dt * omega_c), 0]
    den = [1, -np.exp(-dt * omega_c)]
    # lpf_op = lfilter(num, den, rotspeed_ts)
    time = [k * dt for k in range(len(rotspeed_ts))]
    _, lpf_op = dlti(num, den, dt=dt).output(rotspeed_ts, time, x0=rotspeed_ts[0])

    # cut_idx = int(10 / dt)
    # lpf_op[:cut_idx] = rotspeed_ts[0]

    # fig, ax = plt.subplots(1, 1, figsize=[20.4, 4.8])
    # ax.plot(time, rotspeed_ts, label='Unfiltered')
    # ax.plot(time, lpf_op, label='Filtered')
    # ax.legend()
    # plt.show()

    return lpf_op

def generate_secord_lpf_var(windspeed_ts=None, bts_filename=None, omega=2*np.pi*0.025, zeta=0.707, dt=0.05):
    num = [0, 0, omega**2]
    den = [1, 2 * omega * zeta, omega**2]

    if windspeed_ts is None:
        bts_obj = TurbSimFile(bts_filename)
        windspeed_ts = bts_obj['u'][0, :, 10, 10]
        time = bts_obj['t']
    else:
        time = [k * dt for k in range(len(windspeed_ts))]

    _, lpf_op = lti(num, den).to_discrete(dt).output(windspeed_ts, time, windspeed_ts[0])
    # cut_idx = int(10 / dt)
    # lpf_op[:cut_idx] = windspeed_ts[0]

    # fig, ax = plt.subplots(1, 1, figsize=[20.4, 4.8])
    # ax.plot(time, windspeed_ts, label='Unfiltered')
    # ax.plot(time, lpf_op, label='Filtered')
    # plt.show()
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

def remove_spikes(u, thresh=1, method='linear', n_iter=1, time=None):
    n_datapoints = u.shape[0]

    if time is None:
        time = np.arange(n_datapoints)

    # scale signal
    mean = np.mean(u)
    if mean == 0:
        mean = 1

    u -= mean
    rms = np.sqrt((1 / n_datapoints) * np.sum(u**2))
    if rms == 0:
        rms = 1

    u /= rms

    # identify spikes
    du = (u[1:] - u[:-1]) / rms
    ddu = du[1:] - du[:-1]
    spike_idx = np.where(np.abs(ddu) > thresh)[0] # + 1

    # interpolate over spikes
    i = 0
    while i < len(spike_idx):
        spike_width = 1
        # loop through remaining outlying datapoints after this (i) spike
        for j in range(i + 1, len(spike_idx)):
            # if the spike indices are NOT consecutive, they area not part of this (i) spike
            if spike_idx[j] - spike_idx[j - 1] > 1:
                break
            # otherwise, if they are consecutive, increment the width of this spike
            spike_width += 1

        if spike_idx[i] + spike_width == n_datapoints:
            # spike is at the end of the time series - use last valid value
            u[spike_idx[i]:] = u[spike_idx[i] - 1]
        elif spike_idx[i] == 0:
            # spike is at the beginning of the time series - use first valid vlue
            u[:spike_width] = u[spike_width + 1]
        else:
            # linearly interpolate spikes
            # fetch all datapoints in spike
            spike_ends_idx = np.unique(spike_idx[[i, i + spike_width - 1]])
            spike_all_idx = spike_idx[i] + np.arange(spike_width)
            u[spike_all_idx] = interp1d(
                # fetch time at indices shifted one back and one forward from spike indices
                x=np.concatenate([time[spike_ends_idx - 1], time[spike_ends_idx + 1]]),
                y=np.concatenate([u[spike_ends_idx - 1], u[spike_ends_idx + 1]]),
                kind=method, assume_sorted=False)(time[spike_idx[i] + np.arange(spike_width)])

        i += spike_width

    # generate filtered signal
    u_filt = u * rms + mean

    # run iteratively if desired
    if n_iter > 1:
        u_filt = remove_spikes(u_filt, thresh, method, n_iter-1, time)

    return u_filt