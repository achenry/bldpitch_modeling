import matplotlib.pyplot as plt
import numpy as np
import os
import pysindy as ps
import pickle
import pandas as pd

# ignore user warnings
import warnings
import multiprocessing as mp
from multiprocessing import Pool

from helper_functions import *

RUN_BLDPITCH1_SIM = True
RUN_P_MOTOR_PARTIAL_SIM = True
RELOAD_DATA = False
REPROCESS_DATA = True

warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(1000)  # Seed for reproducibility

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

V_rated = 10
GenSpeed_rated = 4.8 * 2 * np.pi / 60

THRESHOLD = 0.0001
THRESHOLD = 0.001
THRESHOLD = 0.01
THRESHOLD = 0.1
THRESHOLD = 1


def main():
    ## Load Training Data
    vmean_list = list(range(6, 26, 2))  # mean wind speeds in data
    seed_list = list(range(2, 7, 1))  # seeds in data
    n_wind = len(vmean_list)
    n_seed = len(seed_list)
    n_datasets = n_wind * n_seed

    data_dir = '../SOAR-25-V2f_DLC12_PitchActuatorAnalysis'

    model_names = ['BldPitch1', 'P_motor']

    if RELOAD_DATA:
        data_all = load_data(data_dir, vmean_list, seed_list, V_rated, GenSpeed_rated)

        with open(f'./data_all.txt', 'wb') as fh:
            pickle.dump(data_all, fh)

    else:
        with open(f'./data_all.txt', 'rb') as fh:
            data_all = pickle.load(fh)

    # Choose case index for plotting
    vmean_test = 20
    seed_test = 2
    test_idx = [seed_idx for seed_idx in [idx for idx, vmean in enumerate(data_all['vmean']) if vmean == vmean_test]
                if seed_idx in [idx for idx, seed in enumerate(data_all['seed']) if seed == seed_test]][0]

    if REPROCESS_DATA:
        t_train, t_test, u_train, u_test, x_train, x_test, feature_cols, state_cols, ctrl_inpt_cols, n_features \
            = process_data(data_all, n_datasets, model_names)

        for var_name in ['t_train', 't_test', 'u_train', 'u_test', 'x_train', 'x_test', 'feature_cols', 'state_cols',
                         'ctrl_inpt_cols', 'n_features']:
            with open(f'./{var_name}.txt', 'wb') as fh:
                pickle.dump(locals()[var_name], fh)

    else:
        for var_name in ['t_train', 't_test', 'u_train', 'u_test', 'x_train', 'x_test', 'feature_cols', 'state_cols',
                         'ctrl_inpt_cols', 'n_features']:
            with open(f'./{var_name}.txt', 'rb') as fh:
                globals()[var_name] = pickle.load(fh)

        t_train = globals()['t_train']
        t_test = globals()['t_test']
        u_train = globals()['u_train']
        u_test = globals()['u_test']
        x_train = globals()['x_train']
        x_test = globals()['x_test']
        feature_cols = globals()['feature_cols']
        state_cols = globals()['state_cols']
        ctrl_inpt_cols = globals()['ctrl_inpt_cols']
        n_features = globals()['n_features']

    ## Define Candidate Functions
    full_lib = generate_candidate_funcs(model_names, n_features, feature_cols)

    ## Fit the model for discrete-time system

    results_dir = f'./results_thr-{THRESHOLD}'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    fig_dir = f'./figs'
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    dt = t_train[1] - t_train[0]

    optimizer = {key: ps.STLSQ(threshold=THRESHOLD) for key in model_names}

    models = {key: ps.SINDy(
        discrete_time=True,
        optimizer=optimizer[key],
        feature_names=feature_cols[key],
        feature_library=full_lib[key]
    )
        for key in model_names}

    # Check if equal to u_train['P_motor'][test_idx][ctrl_inpt_cols['P_motor'].index('BldPitch1_dot')
    # if not os.path.exists(f'./model_coeffs.txt'):

    ensemble_coeffs = {key: None for key in model_names}
    candidate_funcs = {key: None for key in model_names}
    for m_idx, key in enumerate(model_names):
        # if not os.path.exists(os.path.join(results_dir, f'{key}_ensemble_coeffs.txt')):
        print(f'\nComputing model for {key}=f({[models[key].feature_names]})\n')

        # Ensemble: sub-sample the time series
        models[key].fit(x=x_train[key], t=dt, u=u_train[key], multiple_trajectories=True,
                        ensemble=True)
        # , library_ensemble=True,
        #           n_candidates_to_drop=2)
        ensemble_coeffs[key] = np.mean(models[key].coef_list, axis=0)
        models[key].model.steps[-1][1].optimizer.coef_ = ensemble_coeffs[key]
        models[key].print()

        #     with open(os.path.join(results_dir, f'{key}_ensemble_coeffs.txt'), 'wb') as fh:
        #         pickle.dump(ensemble_coeffs, fh)
        #     with open(os.path.join(results_dir, f'{key}_n_features_in_.txt'), 'wb') as fh:
        #         pickle.dump(full_lib[key].n_features_in_, fh)
        #     with open(os.path.join(results_dir, f'{key}_n_output_features_.txt'), 'wb') as fh:
        #         pickle.dump(full_lib[key].n_output_features_, fh)
        # else:
        #     with open(os.path.join(results_dir, f'{key}_ensemble_coeffs.txt'), 'rb') as fh:
        #         ensemble_coeffs[key] = optimizer[key].coef_  = pickle.load(fh)
        #     with open(os.path.join(results_dir, f'{key}_n_features_in_.txt'), 'rb') as fh:
        #         models[key].feature_library.n_features_in_ = full_lib[key].n_features_in_ = pickle.load(fh)
        #     with open(os.path.join(results_dir, f'{key}_n_output_features_.txt'), 'rb') as fh:
        #         models[key].feature_library.n_output_features_ = full_lib[key].n_output_features_ = pickle.load(fh)

    ## Plot FFT of RootMzc1
    # fft_fig, fft_ax = plot_fft(u_train['BldPitch1'][test_idx][ctrl_inpt_cols['BldPitch1'].index("RootMzc1")], dt, "RootMzc1")
    # fft_fig.show()
    # fft_fig.savefig(os.path.join('./fft_fig.png'))

    ## Simulate and Plot Results
    # if this is the BldPitch1 model, compute the modelled BldPitch1, BldPitch1_dot and BldPitch1_ddot values
    # test_idx = 45
    # res = simulate_case('BldPitch1', test_idx, ctrl_inpt_cols,
    #                     t_train, t_test,
    #                     x_train, x_test,
    #                     u_train, u_test,
    #                     models, ensemble_coeffs, full_lib['BldPitch1'])
    if RUN_BLDPITCH1_SIM:
        if not os.path.exists(os.path.join(results_dir, f'BldPitch1_sim.txt')):
            BldPitch1_sim = []
            for case_idx in range(n_datasets):
                BldPitch1_sim.append(simulate_case('BldPitch1', case_idx, ctrl_inpt_cols,
                                                   t_train, t_test,
                                                   x_train, x_test,
                                                   u_train, u_test,
                                                   models, ensemble_coeffs, full_lib['BldPitch1']))

            with open(os.path.join(results_dir, f'BldPitch1_sim.txt'), 'wb') as fh:
                pickle.dump(BldPitch1_sim, fh)
        else:
            with open(os.path.join(results_dir, f'BldPitch1_sim.txt'), 'rb') as fh:
                BldPitch1_sim = pickle.load(fh)

        # BldPitch1, BldPitch1_dot and BldPitch1_ddot computed from true BldPitch1
        if not os.path.exists(os.path.join(results_dir, f'BldPitch1_true.txt')):

            pool = Pool(processes=mp.cpu_count())
            BldPitch1_true = pool.starmap(generate_bldpitch1_true_drvts,
                                          [(case_idx, u_train, u_test, t_train, t_test, ctrl_inpt_cols)
                                           for case_idx in range(n_datasets)])
            pool.close()

            with open(os.path.join(results_dir, f'BldPitch1_true.txt'), 'wb') as fh:
                pickle.dump(BldPitch1_true, fh)
        else:
            with open(os.path.join(results_dir, f'BldPitch1_true.txt'), 'rb') as fh:
                BldPitch1_true = pickle.load(fh)

        if not os.path.exists(os.path.join(results_dir, f'BldPitch1_model_unfilt.txt')) or \
                not os.path.exists(os.path.join(results_dir, f'BldPitch1_model_filt.txt')):
            pool = Pool(processes=mp.cpu_count())

            # BldPitch1, BldPitch1_dot and BldPitch1_ddot computed from filtered BldPitch1 model
            BldPitch1_model_filt = pool.starmap(generate_BldPitch1_drvts,
                                                [(case_idx, BldPitch1_sim, state_cols, dt, True)
                                                 for case_idx in range(n_datasets)])

            # BldPitch1, BldPitch1_dot and BldPitch1_ddot computed from unfiltered BldPitch1 model
            BldPitch1_model_unfilt = pool.starmap(generate_BldPitch1_drvts,
                                                  [(case_idx, BldPitch1_sim, state_cols, dt, False)
                                                   for case_idx in range(n_datasets)])
            pool.close()

            with open(os.path.join(results_dir, f'BldPitch1_model_unfilt.txt'), 'wb') as fh:
                pickle.dump(BldPitch1_model_unfilt, fh)

            with open(os.path.join(results_dir, f'BldPitch1_model_filt.txt'), 'wb') as fh:
                pickle.dump(BldPitch1_model_filt, fh)
        else:
            with open(os.path.join(results_dir, f'BldPitch1_model_unfilt.txt'), 'rb') as fh:
                BldPitch1_model_unfilt = pickle.load(fh)

            with open(os.path.join(results_dir, f'BldPitch1_model_filt.txt'), 'rb') as fh:
                BldPitch1_model_filt = pickle.load(fh)

        # Plot BldPitch and derivatives, from true vs from modeled, filtered vs unfiltered for single test case
        BldPitch1_drvts_ts_fig, BldPitch1_drvts_ts_axs \
            = plot_bldpitch_drvts_ts(test_idx, BldPitch1_model_unfilt, BldPitch1_model_filt, BldPitch1_true)
        BldPitch1_drvts_ts_fig.show()

        # Plot BldPitch and derivatives, from true vs modeled, filtered vs unfiltered MSE vs all test_cases
        BldPitch1_drvts_mse_fig, BldPitch1_drvts_mse_axs, BldPitch1_drvts_mse \
            = plot_bldpitch1_mse(n_datasets, u_train, u_test, ctrl_inpt_cols, BldPitch1_model_filt,
                                 BldPitch1_model_unfilt)
        BldPitch1_drvts_mse_fig.show()
        BldPitch1_drvts_mse.to_csv(os.path.join(results_dir, f'BldPitch1_drvts_mse.csv'))

        if not os.path.exists(os.path.join(results_dir, f'P_motor_full_sim.txt')):
            P_motor_full_sim = []
            for case_idx in range(n_datasets):
                # Using derivatives based on filtered BldPitch1
                P_motor_full_sim.append(simulate_case('P_motor', case_idx, ctrl_inpt_cols,
                                                      t_train, t_test,
                                                      x_train, x_test,
                                                      u_train, u_test,
                                                      models, ensemble_coeffs, full_lib['P_motor'],
                                                      True, BldPitch1_model_filt))
            with open(os.path.join(results_dir, f'P_motor_full_sim.txt'), 'wb') as fh:
                pickle.dump(P_motor_full_sim, fh)
        else:
            with open(os.path.join(results_dir, f'P_motor_full_sim.txt'), 'rb') as fh:
                P_motor_full_sim = pickle.load(fh)

    if RUN_P_MOTOR_PARTIAL_SIM:
        if not os.path.exists(os.path.join(results_dir, f'P_motor_partial_sim.txt')):
            P_motor_partial_sim = []
            for case_idx in range(n_datasets):
                P_motor_partial_sim.append(simulate_case('P_motor', case_idx, ctrl_inpt_cols,
                                                         t_train, t_test,
                                                         x_train, x_test,
                                                         u_train, u_test,
                                                         models, ensemble_coeffs, full_lib['P_motor'],
                                                         False, None))
            with open(os.path.join(results_dir, 'P_motor_partial_sim.txt'), 'wb') as fh:
                pickle.dump(P_motor_partial_sim, fh)
        else:
            with open(os.path.join(results_dir, 'P_motor_partial_sim.txt'), 'rb') as fh:
                P_motor_partial_sim = pickle.load(fh)

    if RUN_P_MOTOR_PARTIAL_SIM and RUN_BLDPITCH1_SIM:
        # compute modelled P_motor mean for each seed and vmean for partial and full P_motor model
        # for each mean wind speed
        P_motor_stats = {'seed': data_all['seed'], 'vmean': data_all['vmean'],
                         'mean': {'seed_true': data_all['P_motor_mean_seed'],
                                  'vmean_true': data_all['P_motor_mean_vmean'],
                                  'seed_partial': [],
                                  'vmean_partial': [],
                                  'seed_full': [],
                                  'vmean_full': [],
                                  },
                         'max': {'seed_true': data_all['P_motor_max_seed'],
                                 'vmean_true': data_all['P_motor_max_vmean'],
                                 'seed_partial': [],
                                 'vmean_partial': [],
                                 'seed_full': [],
                                 'vmean_full': [],
                                 }
                         }

        P_motor_mse = {'P_motor_full': [], 'P_motor_partial': []}
        for v_idx in range(len(vmean_list)):
            # for each turbulence seed
            for s_idx in range(len(seed_list)):
                case_idx = (v_idx * n_seed) + s_idx
                print(f'\nComputing P_motor Mean and Max for Case {case_idx}...')

                _, x_sim_full, _, _, _ = P_motor_partial_sim[case_idx]
                P_motor_model_partial = x_sim_full[:, state_cols['P_motor'].index('P_motor')]

                _, x_sim_partial, _, _, _ = P_motor_full_sim[case_idx]
                P_motor_model_full = x_sim_partial[:, state_cols['P_motor'].index('P_motor')]

                # compute P_motor_mean and max for this seed and vmean for partial model
                P_motor_stats['mean']['seed_partial'].append(np.mean(P_motor_model_partial))
                P_motor_stats['mean']['seed_full'].append(np.mean(P_motor_model_full))
                P_motor_stats['max']['seed_partial'].append(np.max(P_motor_model_partial))
                P_motor_stats['max']['seed_full'].append(np.max(P_motor_model_full))

                # Compute MSE between true and modeled values for each case
                P_motor_mse['P_motor_full'].append(compute_mse(
                    data_true=np.concatenate(
                        [x_train['P_motor'][case_idx][state_cols['P_motor'].index('P_motor')],
                         x_test['P_motor'][case_idx][state_cols['P_motor'].index('P_motor')]]),
                    data_modeled=P_motor_model_full))

                P_motor_mse['P_motor_partial'].append(compute_mse(
                    data_true=np.concatenate(
                        [x_train['P_motor'][case_idx][state_cols['P_motor'].index('P_motor')],
                         x_test['P_motor'][case_idx][state_cols['P_motor'].index('P_motor')]]),
                    data_modeled=P_motor_model_partial))

            # compute P_motor_mean and max over all seeds for this vmean
            for metric in ['mean', 'max']:
                for model_type in ['partial', 'full']:
                    if metric == 'mean':
                        new_vals = [np.mean([P_val for vmean, P_val in
                                      zip(P_motor_stats['vmean'], P_motor_stats[metric][f'seed_{model_type}']) if
                                      vmean == vmean_list[v_idx]])] * n_seed
                    else:
                        new_vals = [np.max([P_val for vmean, P_val in
                                      zip(P_motor_stats['vmean'], P_motor_stats[metric][f'seed_{model_type}']) if
                                      vmean == vmean_list[v_idx]])] * n_seed
                    P_motor_stats[metric][f'vmean_{model_type}'] \
                        = P_motor_stats[metric][f'vmean_{model_type}'] \
                          + new_vals

        # Plot P_motor_partial and P_motor_full MSE vs all test_cases
        P_motor_mse_fig, P_motor_mse_axs = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
        width = 0.25
        ind = np.array(range(1, n_datasets + 1))
        P_motor_mse_axs[0].bar(ind, P_motor_mse['P_motor_partial'], width, color='r')
        P_motor_mse_axs[0].set(ylabel='P_motor Partial MSE')
        P_motor_mse_axs[1].bar(ind, P_motor_mse['P_motor_full'], width, color='r')
        P_motor_mse_axs[1].set(ylabel='P_motor Full MSE')
        P_motor_mse_axs[-1].set(xlabel='Case Number')
        P_motor_mse_fig.show()
        P_motor_mse_fig.savefig(os.path.join(fig_dir, f'pmotor_mse_fig_thr-{THRESHOLD}.png'))

    # setup figures
    t_true = np.concatenate([t_train, t_test])
    if RUN_BLDPITCH1_SIM and RUN_P_MOTOR_PARTIAL_SIM:
        coeff_fig, coeff_axs = plt.subplots(len(model_names), 1, figsize=FIGSIZE)
        sim_ts_fig, sim_ts_axs = plt.subplots(len(model_names), 1, sharex=True, figsize=FIGSIZE)
        traj_labels = {'P_motor': ["BldPitch1"], 'BldPitch1': ["Wind1VelX"]}
        P_mean_fig, P_mean_axs = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
        ensemble_fig, ensemble_axs = plt.subplots(len(model_names), 1, figsize=FIGSIZE)

        for m_idx, key in enumerate(model_names):

            op_idx = state_cols[key].index(key)
            ensemble_coeffs_df = plot_ensemble_error(models, key, m_idx, state_cols, op_idx, ensemble_axs)
            ensemble_coeffs_df.to_csv(os.path.join(results_dir, f'{key}_ensemble_coefficients.csv'))

            if key == 'BldPitch1':
                t_sim, x_sim, _, _, _ = BldPitch1_sim[test_idx]
            elif key == 'P_motor':
                t_sim, x_sim, _, _, _ = P_motor_partial_sim[test_idx]
                t_sim_full, x_sim_full, _, _, _ = P_motor_full_sim[test_idx]

                # Plot full P_motor model simulation (using P_motor and BldPitch1 models)
                # plot_time_series(key, test_idx, m_idx, op_idx, t_train, t_sim_full, t_true, x_train, x_test, x_sim_full,
                #                  sim_ts_axs, full_model=True)

                # Plot barchart of true vs partial vs full P_mean for each vmean
                width = 0.25
                ind = np.array(vmean_list)
                P_mean_axs[0].bar(ind, P_motor_stats['mean']['vmean_true'][::n_seed], width, color='r', label="True")
                P_mean_axs[0].bar(ind + width, P_motor_stats['mean']['vmean_partial'][::n_seed], width, color='g',
                                  label="Partial Model")
                P_mean_axs[1].bar(ind, P_motor_stats['mean']['vmean_true'][::n_seed], width, color='r', label="True")
                P_mean_axs[1].bar(ind + width, P_motor_stats['mean']['vmean_full'][::n_seed], width, color='g',
                                  label="Partial Model")
                # P_mean_ax.bar(ind + 2 * width, P_motor_mean['vmean_full'][::n_seed], width, color='b', label="Full Model")
                P_mean_axs[0].set(xticks=vmean_list, xlabel="Mean Wind Speed, $\\bar{V}$")
                P_mean_axs[1].set(ylabel="Mean Motor Power, $\\bar{P}_{motor}$")
                P_mean_axs[0].legend()

            # Plot barchart of coefficients for different terms for BlPitch1 and P_motor
            coeffs_df = plot_coeffs(models, key, m_idx, op_idx, coeff_axs)
            coeffs_df.to_csv(os.path.join(results_dir, f'{key}_coefficients.csv'))

            # Plot Simulated, True Pmotor, Beta vs time
            plot_time_series(key, test_idx, m_idx, op_idx, t_train, t_sim, t_true, x_train, x_test, x_sim,
                             sim_ts_axs, full_model=False)

            # Plot Pmotor vs Beta, Beta vs V,
            # plot_correlations(key, test_idx, m_idx, op_idx, traj_axs, traj_labels, ctrl_inpt_cols, x_sim_train, x_sim_test,
            #                   x_train, x_test, u_train, u_test)

        coeff_fig.show()
        sim_ts_fig.show()
        # traj_fig.show()
        P_mean_fig.show()
        ensemble_fig.show()

        ## Save Results
        coeff_fig.savefig(os.path.join(fig_dir, f'coeff_fig_thr-{THRESHOLD}.png'))
        sim_ts_fig.savefig(os.path.join(fig_dir, f'sim_ts_fig_thr-{THRESHOLD}.png'))
        # traj_fig.savefig(os.path.join(results_dir, 'traj_fig.png'))
        P_mean_fig.savefig(os.path.join(fig_dir, f'P_mean_fig_thr-{THRESHOLD}.png'))
        ensemble_fig.savefig(os.path.join(fig_dir, f'ensemble_fig_thr-{THRESHOLD}.png'))

    BldPitch1_drvts_ts_fig.savefig(os.path.join(fig_dir, f'bldpitch1_sim_ts_fig_thr-{THRESHOLD}.png'))
    BldPitch1_drvts_mse_fig.savefig(os.path.join(fig_dir, f'bldpitch1_mse_fig_thr-{THRESHOLD}.png'))

    # for each metric, each case (every seed of every wind speed) and each vmean, each model type, \
    # compute mse to compare across threshold values

    P_motor_mse = {
        'mean': {'seed_partial': None,
                 'vmean_partial': None,
                 'seed_full': None,
                 'vmean_full': None,
                 },
        'max': {'seed_partial': None,
                'vmean_partial': None,
                'seed_full': None,
                'vmean_full': None,
                }
    }
    for metric in ['mean', 'max']:
        for group in ['seed', 'vmean']:
            for model_type in ['partial', 'full']:
                P_motor_mse[metric][f'{group}_{model_type}_mse'] \
                    = compute_mse(P_motor_stats[metric][f'{group}_true'],
                                  P_motor_stats[metric][f'{group}_{model_type}'])

    # pd.DataFrame(P_motor_stats).to_csv(os.path.join(results_dir, f'P_motor_stats.csv'))
    with open(os.path.join(results_dir, 'P_motor_mse.txt'), 'wb') as fh:
        pickle.dump(P_motor_mse, fh)

    COMPARE_THR = True
    if COMPARE_THR:
        thresholds = [0.0001, 0.001, 0.01, 0.1, 1]

        BldPitch1_stats_mse = {'seed': [], 'vmean': []}

        for thr in thresholds:
            results_dir = f'./results_thr-{thr}'
            BldPitch1_mse = pd.read_csv(os.path.join(results_dir, 'BldPitch1_drvts_mse.csv'))
            BldPitch1_mse = BldPitch1_mse.rename(columns={'BldPitch1_unfilt': 'seed'})
            BldPitch1_mse = BldPitch1_mse.drop(columns=[col for col in BldPitch1_mse.columns if col != 'seed'])
            vmean_agg = BldPitch1_mse['seed'].groupby(BldPitch1_mse.index // n_seed).agg('mean')
            BldPitch1_mse['vmean'] = vmean_agg.loc[vmean_agg.index.repeat(n_seed)].reset_index(drop=True)

            for group in ['seed', 'vmean']:
                BldPitch1_stats_mse[group].append(BldPitch1_mse[group].mean())

        BldPitch1_mse_fig, BldPitch1_mse_ax = plt.subplots(1, 1, figsize=FIGSIZE, sharex=True)
        width = 0.25
        ind = np.array(np.log10(thresholds))
        # Just testing Bldpitch etc to Pmotor - first row
        # Testing all vars and Bldpitch model to Pmotor - second row
        # Pmean - first col, Pmax - second col
        BldPitch1_mse_ax.bar(ind, BldPitch1_stats_mse[f'seed'], width, color='r')
        BldPitch1_mse_ax.set(ylabel=f"BldPitch1")
        BldPitch1_mse_ax.set(title=f"MSE of BldPitch1 Modeled vs True Data")
        BldPitch1_mse_ax.set(xticks=ind, xlabel="Threshold, $\\lambda$",
                             xticklabels=[f'10^{int(i)}' for i in ind])

        BldPitch1_mse_fig.show()
        BldPitch1_mse_fig.savefig(os.path.join(fig_dir, f'BldPitch1_thr_mse.png'))

        P_motor_stats_mse = {'mean': {'seed_partial': [], 'vmean_partial': [], 'seed_full': [], 'vmean_full': []},
                             'max': {'seed_partial': [], 'vmean_partial': [], 'seed_full': [], 'vmean_full': []}}

        for thr in thresholds:
            results_dir = f'./results_thr-{thr}'
            with open(os.path.join(results_dir, 'P_motor_mse.txt'), 'rb') as fh:
                P_motor_mse = pickle.load(fh)

            for metric in ['mean', 'max']:
                for group in ['seed', 'vmean']:
                    for model_type in ['partial', 'full']:
                        P_motor_stats_mse[metric][f'{group}_{model_type}'].append(
                            P_motor_mse[metric][f'{group}_{model_type}_mse'])

        P_mse_fig, P_mse_axs = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True)
        width = 0.25
        ind = np.array(np.log10(thresholds))
        # Just testing Bldpitch etc to Pmotor - first row
        # Testing all vars and Bldpitch model to Pmotor - second row
        # Pmean - first col, Pmax - second col
        for model_type_idx, model_type in enumerate(['partial', 'full']):
            for metric_idx, metric in enumerate(['mean', 'max']):
                P_mse_axs[model_type_idx][metric_idx].bar(ind, P_motor_stats_mse[metric][f'seed_{model_type}'], width,
                                                          color='r',
                                                          label=f"{model_type.capitalize()} Model for each Seed")
                P_mse_axs[model_type_idx][metric_idx].bar(ind + width, P_motor_stats_mse[metric][f'vmean_{model_type}'],
                                                          width, color='g',
                                                          label=f"{model_type.capitalize()} Model for each Wind Speed")
                P_mse_axs[model_type_idx][metric_idx].set(ylabel=f"{metric.capitalize()} P_motor")
                P_mse_axs[0][metric_idx].set(title=f"MSE of {metric.capitalize()} Motor Power Modeled vs True Data")
                P_mse_axs[-1][metric_idx].set(xticks=ind, xlabel="Threshold, $\\lambda$",
                                              xticklabels=[f'10^{int(i)}' for i in ind])
                # P_mse_axs[-1][metric_idx].set_xscale('log')
                # P_mse_axs[0][metric_idx].set_xscale('log')

            P_mse_axs[model_type_idx][0].legend()

        P_mse_fig.show()
        P_mse_fig.savefig(os.path.join(fig_dir, f'P_thr_mse.png'))

        # generate csv of coefficients (mean and standard deviation) for different values of lambda
        # all_coeffs
        all_coeffs_df = {key: pd.DataFrame(data={}, index=ensemble_coeffs_df['Candidate Function']) for key in model_names}
        for key in model_names:
            for thr in thresholds:
                results_dir = f'./results_thr-{thr}'
                thr_coeffs_df = pd.read_csv(os.path.join(results_dir, f'{key}_ensemble_coefficients.csv'))

                thr_coeffs_df = thr_coeffs_df.set_index('Candidate Function')
                thr_coeffs_df.rename(columns={'Mean': f'mean_{thr}', 'Standard Deviation': f'std_{thr}'}, inplace=True)
                thr_coeffs_df = thr_coeffs_df[[f'mean_{thr}', f'std_{thr}']]
                all_coeffs_df[key] = pd.merge(all_coeffs_df[key], thr_coeffs_df, left_index=True, right_index=True,
                                              how="outer")

            all_coeffs_df[key].to_csv(os.path.join('./', f'all_{key}_coefficients.csv'))


if __name__ == '__main__':
    main()
