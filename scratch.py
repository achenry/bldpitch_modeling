# from helper_functions import generate_secord_lpf_var, generate_firstord_lpf_var
#
# generate_secord_lpf_var(bts_filename='/Users/aoifework/Documents/Research/ipc_tuning/TurbSim/TurbSim_IF/TurbSim.bts')

# generate_firstord_lpf_var()

# pass

# beta_dot1 = [
#     (data_dict[f'BldPitch{b + 1}'][init_idx + 1:] - data_dict[f'BldPitch{b + 1}'][init_idx - 1:-2]) \
#     / (2 * dt) for b in range(3)]
# beta_ddot1 = [
#     (data_dict[f'BldPitch{b + 1}'][init_idx + 1:] - 2 * data_dict[f'BldPitch{b + 1}'][init_idx:-1]
#      + data_dict[f'BldPitch{b + 1}'][init_idx - 1:-2]) / (2 * dt) for b in range(3)]


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
import numpy as np
r = 2
constraint_zeros = np.zeros(int(r * (r + 1) / 2))
constraint_matrix = np.zeros((int(r * (r + 1) / 2),
                              int(r * (r ** 2) / 2)))
q = r
for i in range(r):
    constraint_matrix[i, i * (r + 1)] = 1.0
    counter = 1
    for j in range(i + 1, r):
        constraint_matrix[q, i * r + j] = 1.0
        constraint_matrix[q, i * r + j + counter * (r - 1)] = 1.0
        counter += 1
        q += 1

pass
# compute filtered BldPitch1 modeled signal
        if not os.path.exists(os.path.join(results_dir, f'BldPitch1_model_unfilt.txt')) or \
            not os.path.exists(os.path.join(results_dir, f'BldPitch1_model_filt.txt')):
            pool = Pool(processes=mp.cpu_count())

            # BldPitch1, BldPitch1_dot and BldPitch1_ddot computed from filtered BldPitch1 model
            BldPitch1_model_filt = pool.starmap(generate_drvts,
                                                [('BldPitch1', case_idx, models, simulator_cases,
                                                  data_processor.state_cols, dt, True)
                                                 if case_idx in test_indices else None for case_idx in
                                                 range(data_processor.n_datasets)])

            # BldPitch1, BldPitch1_dot and BldPitch1_ddot computed from unfiltered BldPitch1 model
            BldPitch1_model_unfilt = pool.starmap(generate_drvts,
                                                  [('BldPitch1', case_idx, models, simulator_cases,
                                                    data_processor.state_cols, dt, False)
                                                   if case_idx in test_indices else None for case_idx in
                                                   range(data_processor.n_datasets)])
            pool.close()

            for model in ['unfilt', 'filt']:
                with open(os.path.join(results_dir, f'BldPitch1_model_{model}.txt'), 'wb') as fh:
                    pickle.dump(locals()[f'BldPitch1_model_{model}'], fh)

        else:
            with open(os.path.join(results_dir, f'BldPitch1_model_unfilt.txt'), 'rb') as fh:
                BldPitch1_model_unfilt = pickle.load(fh)

            with open(os.path.join(results_dir, f'BldPitch1_model_filt.txt'), 'rb') as fh:
                BldPitch1_model_filt = pickle.load(fh)

# Plot BldPitch and derivatives, from true vs from modeled, filtered vs unfiltered for single test case
BldPitch1_drvts_true = generate_bldpitch1_true_drvts(test_indices[0], data_processor)
BldPitch1_drvts_ts_fig, BldPitch1_drvts_ts_axs \
    = plot_bldpitch_drvts_ts(test_indices[0], data_processor, BldPitch1_drvts_true)
BldPitch1_drvts_ts_fig.show()

# ['BldPitch1',
#  'BldPitch1_dot', USE
#  'BldPitch1_ddot',
#  'BldPitch1^2',
#  'BldPitch1 BldPitch1_dot', USE X
#  'BldPitch1 BldPitch1_ddot',
#  'BldPitch1_dot^2', USE X
#  'BldPitch1_dot BldPitch1_ddot', USE
#  'BldPitch1_ddot^2']

# zero_coeff_indices = [0, 2, 3, 4, 6, 7, 8]# [0, 2, 3, 5, 8]
# n_constraints = len(zero_coeff_indices)
# initial_guess = np.zeros((n_states, n_features))
# constraint_lhs = np.zeros((n_constraints, n_features))
# constraint_rhs = np.zeros(n_constraints)
# for cons_idx, coeff_idx in enumerate(zero_coeff_indices):
#     constraint_lhs[cons_idx, coeff_idx] = 1
# # initial_guess
# optimizer[key] = ps.ConstrainedSR3(
#                                    threshold=THRESHOLD,
#                                    thresholder='l1',
#                                    constraint_order='target',
#                                    constraint_lhs=constraint_lhs,
#                                    constraint_rhs=constraint_rhs,
#                                    # nu=1e6,
#                                    max_iter=1000
# )
# optimizer[key] = ps.STLSQ(threshold=THRESHOLD)
# optimizer[key] = ps.SSR(alpha=THRESHOLD)

# n_features = 113
# n_states = len(data_processor.state_cols[key])
# # constraint_lhs = np.zeros((1, n_features * n_states))
# # constraint_rhs = np.zeros(1)
#
# # constraint_lhs[0, 2] = 1
# initial_guess = np.zeros((n_states, n_features))
# initial_guess[0, 2] = 1
# # constraint_lhs[0, 2] = 1


# optimizer[key] = ps.ConstrainedSR3(threshold=THRESHOLD[key],
# thresholder='l0',
# constraint_lhs=constraint_lhs,
# constraint_rhs=constraint_rhs,
# initial_guess=initial_guess,
# normalize_columns=True,
# max_iter=100
# )
