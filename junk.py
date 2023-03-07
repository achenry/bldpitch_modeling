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

# all_libs['BldPitch1_dot'].append(poly_lib2['BldPitch1_dot'])
# idx = np.array([1] + list(range(1, n_features['BldPitch1_dot'])))
# idx[feature_cols['BldPitch1_dot'].index('GenTq')] = feature_cols['BldPitch1_dot'].index('Wind1VelX')
# idx[feature_cols['BldPitch1_dot'].index('GenSpeed_dev_rated')] = feature_cols['BldPitch1_dot'].index('Wind1VelX')
# idx[feature_cols['BldPitch1_dot'].index('Wind1VelX_dev_rated')] = feature_cols['BldPitch1_dot'].index('Wind1VelX')
# idx[feature_cols['BldPitch1_dot'].index('Wind1VelX_dev')] = feature_cols['BldPitch1_dot'].index('Wind1VelX')
# idx[feature_cols['BldPitch1_dot'].index('Azimuth')] = feature_cols['BldPitch1_dot'].index('Wind1VelX')
# inputs_per_library['BldPitch1_dot'] = np.vstack([inputs_per_library['BldPitch1_dot'], idx])
#
# all_libs['BldPitch1_ddot'].append(poly_lib2['BldPitch1_ddot'])
# idx = np.array([1] + list(range(1, n_features['BldPitch1_ddot'])))
# idx[feature_cols['BldPitch1_ddot'].index('GenTq')] = feature_cols['BldPitch1_ddot'].index('Wind1VelX')
# idx[feature_cols['BldPitch1_ddot'].index('GenSpeed_dev_rated')] = feature_cols['BldPitch1_ddot'].index('Wind1VelX')
# idx[feature_cols['BldPitch1_ddot'].index('Wind1VelX_dev_rated')] = feature_cols['BldPitch1_ddot'].index('Wind1VelX')
# idx[feature_cols['BldPitch1_ddot'].index('Wind1VelX_dev')] = feature_cols['BldPitch1_ddot'].index('Wind1VelX')
# idx[feature_cols['BldPitch1_ddot'].index('Azimuth')] = feature_cols['BldPitch1_ddot'].index('Wind1VelX')
# inputs_per_library['BldPitch1_ddot'] = np.vstack([inputs_per_library['BldPitch1_ddot'], idx])

# RotSpeed terms
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

# genspeed_idx = {key: [feature_cols[key].index('GenSpeed')] for key in model_names}
# azimuth_idx = {key: [feature_cols[key].index('Azimuth')] for key in model_names}
# inputs_per_library['P_motor'] = np.vstack([inputs_per_library['P_motor'],
#                                            ((genspeed_idx['P_motor'] + azimuth_idx['P_motor']) * n_features['P_motor'])
#                                            [:n_features['P_motor']]])


# inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'],
#                                              ((genspeed_idx['BldPitch1'] + azimuth_idx['BldPitch1'])
#                                               * n_features['BldPitch1'])[:n_features['BldPitch1']]])

# all_libs['BldPitch1_dot'].append(sin_lib['BldPitch1_dot'])
# inputs_per_library['BldPitch1_dot'] = np.vstack([inputs_per_library['BldPitch1_dot'],
#                                              [1] + list(range(1, n_features['BldPitch1_dot']))])
# inputs_per_library['BldPitch1_dot'] = np.vstack([inputs_per_library['BldPitch1_dot'],
#                                                  ((genspeed_idx['BldPitch1_ddot'] + azimuth_idx['BldPitch1_ddot'])
#                                                   * n_features['BldPitch1_dot'])[:n_features['BldPitch1_dot']]])

# all_libs['BldPitch1_ddot'].append(sin_lib['BldPitch1_ddot'])
# inputs_per_library['BldPitch1_ddot'] = np.vstack([inputs_per_library['BldPitch1_ddot'],
#                                              [1] + list(range(1, n_features['BldPitch1_ddot']))])
# inputs_per_library['BldPitch1_ddot'] = np.vstack([inputs_per_library['BldPitch1_ddot'],
#                                                   ((genspeed_idx['BldPitch1_ddot'] + azimuth_idx['BldPitch1_ddot'])
#                                                    * n_features['BldPitch1_ddot'])[:n_features['BldPitch1_ddot']]])

# inputs_per_library['BldPitch1_dot'] = np.vstack([inputs_per_library['BldPitch1_dot'],
#                                                  (v_dev_rated_idx['BldPitch1_dot'] * n_features['BldPitch1_dot'])[
#                                                  :n_features['BldPitch1_dot']]])
# inputs_per_library['BldPitch1_ddot'] = np.vstack([inputs_per_library['BldPitch1_ddot'],
#                                                   (v_dev_rated_idx['BldPitch1_ddot'] * n_features[
#                                                       'BldPitch1_ddot'])[
#                                                   :n_features['BldPitch1_ddot']]])

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
# fourier_lib = {key: FourierLibrary(n_frequencies=2, include_cos=True, include_sin=True)
#                for key in ['BldPitch1', 'BldPitch1_dot', 'BldPitch1_ddot']}
# all_libs['BldPitch1'].append(fourier_lib['BldPitch1'])
# all_libs['BldPitch1_dot'].append(fourier_lib['BldPitch1_dot'])
# all_libs['BldPitch1_ddot'].append(fourier_lib['BldPitch1_ddot'])
# inputs_per_library['BldPitch1'] = np.vstack([inputs_per_library['BldPitch1'],
#                                              [1] + list(range(1, n_features['BldPitch1']))])
# inputs_per_library['BldPitch1_dot'] = np.vstack([inputs_per_library['BldPitch1_dot'],
#                                              [1] + list(range(1, n_features['BldPitch1_dot']))])
# inputs_per_library['BldPitch1_ddot'] = np.vstack([inputs_per_library['BldPitch1_ddot'],
#                                              [1] + list(range(1, n_features['BldPitch1_ddot']))])

## Candidate funcs

# lib_funcs_1 = [lambda x: x]
# lib_funcs_2 = [lambda x, y: x * y]
#
# lib_func_names_1 = [lambda x: f"{x}"]
# lib_func_names_2 = [lambda x, y: f"{x} {y}"]
#
# inc_indices_1 = pmotor_idx
# inc_indices_1 = beta_dot_idx# + pmotor_idx
# inc_indices_2 = beta_dot_idx + beta_ddot_idx
#
# all_libs['P_motor_partial'].append(
#     CustomLibrary(library_functions=lib_funcs_1, function_names=lib_func_names_1))
# all_libs['P_motor_partial'].append(
#     CustomLibrary(library_functions=lib_funcs_2, function_names=lib_func_names_2))
#
# inputs_per_library['P_motor_partial'] = np.vstack([inputs_per_library['P_motor_partial'],
#                                                    (inc_indices_1
#                                                     * self.n_features['P_motor_partial'])[
#                                                    :self.n_features['P_motor_partial']]])
#
# inputs_per_library['P_motor_partial'] = np.vstack([inputs_per_library['P_motor_partial'],
#                                                    (inc_indices_2
#                                                     * self.n_features['P_motor_partial'])[
#                                                    :self.n_features['P_motor_partial']]])

# s = 100
# fourier_lib_funcs = [
#     lambda x: s * np.sin(1 * x / GENSPEED_SCALING),
#     lambda x: s * np.cos(1 * x / GENSPEED_SCALING),
#     lambda x: s * np.sin(1 * x / GENSPEED_SCALING)**2,
#     lambda x: s * np.cos(1 * x / GENSPEED_SCALING)**2,
#     lambda x: s * np.sin(1 * x / GENSPEED_SCALING) * np.cos(x / GENSPEED_SCALING)
#  ]
#
#
# fourier_lib_func_names = [
#     lambda x: f'sin(1 {x})',
#     lambda x: f'cos(1 {x})',
#     lambda x: f'sin(1 {x})^2',
#     lambda x: f'cos(1 {x})^2',
#     lambda x: f'cos(1 {x}) sin(1 {x}',
# ]
#
# fourier_lib = CustomLibrary(fourier_lib_funcs, fourier_lib_func_names)
# fourier_lib = PolynomialLibrary(degree=1, include_interaction=True, include_bias=False) # CHANGE 5: 1 to 2, only k=1 terms
# cos_genspeed_dev_rated_idx = [self.feature_cols[key].index('cos(GenSpeed_dev_rated)')]
# sin_genspeed_dev_rated_idx = [self.feature_cols[key].index('sin(GenSpeed_dev_rated)')]
# cos_azimuth_idx = [self.feature_cols[key].index('cos(Azimuth)')]
# sin_azimuth_idx = [self.feature_cols[key].index('sin(Azimuth)')]
# genspeed_idx = [self.feature_cols[key].index('GenSpeed')]
# # sinus_indices = [self.feature_cols[key].index(field) for field in np.concatenate([[f'sin({k} GenSpeed)', f'cos({k} GenSpeed)'] for k in range(1, 2)])]


# lambda x: (np.maximum(np.zeros_like(x), x)) ** (1 / 2),
# lambda x: (np.maximum(np.zeros_like(x), x)) ** (1 / 3),
# lambda x: np.maximum(np.zeros_like(x), x)]

# Define inputs for each library
# n_libraries = {key: len(libs) for key, libs in all_libs.items()}


# BldPitch1_dot_sim.append(simulate_case('BldPitch1_dot', case_idx, ctrl_inpt_cols,
#                                    data_processor.t_train, t_test,
#                                    x_train, x_test,
#                                    u_train, u_test,
#                                    models, ensemble_coeffs, full_lib['BldPitch1_dot']))
#
# BldPitch1_ddot_sim.append(simulate_case('BldPitch1_ddot', case_idx, ctrl_inpt_cols,
#                                    data_processor.t_train, t_test,
#                                    x_train, x_test,
#                                    u_train, u_test,
#                                    models, ensemble_coeffs, full_lib['BldPitch1_ddot']))


# for var_name in ['t_train', 't_test', 'u_train', 'u_test', 'x_train', 'x_dot_train', 'x_test', 'feature_cols',
#                  'state_cols', 'state_drvt_vols', 'ctrl_inpt_cols', 'n_features']:
#     with open(f'./{var_name}.txt', 'rb') as fh:
#         if var_name in ['feature_cols', 'n_features']:
#             setattr(data_processor, var_name, pickle.load(fh))
#         else:
#             locals()[var_name] = pickle.load(fh)

# t_train = globals()['t_train']
# t_test = globals()['t_test']
# u_train = globals()['u_train']
# u_test = globals()['u_test']
# x_train = globals()['x_train']
# x_dot_train = globals()['x_dot_train']
# x_test = globals()['x_test']
# # data_processor.feature_cols = globals()['feature_cols']
# state_cols = globals()['state_cols']
# state_drvt_vols = globals()['state_drvt_vols']
# ctrl_inpt_cols = globals()['ctrl_inpt_cols']
# data_processor.n_features = globals()['n_features']

# for var_name in ['t_train', 't_test', 'u_train', 'u_test', 'x_train', 'x_dot_train', 'x_test', 'feature_cols',
#                  'state_cols', 'state_drvt_vols', 'ctrl_inpt_cols', 'n_features']:
#     with open(f'./{var_name}.txt', 'wb') as fh:
#         pickle.dump(locals()[var_name], fh)

# t_train, t_test, u_train, u_test, x_train, x_dot_train, x_test, feature_cols, state_cols, state_drvt_vols, \
# ctrl_inpt_cols, n_features = \

# DONE plot errorbar of mean coeff * candidate terms over all datasets
# DONE try generating model for each wind speed

# BldPitch1, BldPitch1_dot and BldPitch1_ddot computed from true BldPitch1
# if not os.path.exists(os.path.join(results_dir, f'BldPitch1_drvts_true.txt')):
#
#     pool = Pool(processes=mp.cpu_count())
#     BldPitch1_drvts_true = pool.starmap(generate_bldpitch1_true_drvts,
#                                   [(case_idx, u_train, u_test, data_processor.t_train, t_test, ctrl_inpt_cols)
#                                    for case_idx in range(n_datasets)])
#     pool.close()
#
#     with open(os.path.join(results_dir, f'BldPitch1_drvts_true.txt'), 'wb') as fh:
#         pickle.dump(BldPitch1_drvts_true, fh)
# else:
#     with open(os.path.join(results_dir, f'BldPitch1_drvts_true.txt'), 'rb') as fh:
#         BldPitch1_drvts_true = pickle.load(fh)

# [lambda x: f"max(0, {x})^(1/2)",
#                     lambda x: f"max(0, {x})^(1/3)",
#                     lambda x: f"max(0, {x})"]
