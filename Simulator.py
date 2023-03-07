import numpy as np
import os
from sklearn.linear_model._base import safe_sparse_dot
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pickle
import multiprocessing as mp
from multiprocessing import Pool


class Simulator:
    def __init__(self, **kwargs):
        self.model_name = kwargs['model_name']
        self.data_processor = kwargs['data_processor']
        self.model = kwargs['model']

    def simulate_case(self, test_idx, is_discrete=False, is_dynamical=True):
        print(f'\nSimulating {self.model_name} for case {test_idx}')

        # full_lib.library_ensemble = False

        if test_idx in self.data_processor.training_idx:
            dataset_type_idx = np.argwhere(self.data_processor.training_idx == test_idx).squeeze()
            x = self.data_processor.x_train[self.model_name][dataset_type_idx]
            x_dot = self.data_processor.x_dot_train[self.model_name][dataset_type_idx]
            u = self.data_processor.u_train[self.model_name][dataset_type_idx]
            t = self.data_processor.t_train
        elif test_idx in self.data_processor.testing_idx:
            dataset_type_idx = np.argwhere(self.data_processor.testing_idx == test_idx).squeeze()
            x = self.data_processor.x_test[self.model_name][dataset_type_idx]
            x_dot = self.data_processor.x_dot_test[self.model_name][dataset_type_idx]
            u = self.data_processor.u_test[self.model_name][dataset_type_idx]
            t = self.data_processor.t_test

        u_dash = u
        x0 = x[0, :]

        if is_dynamical:
            if not is_discrete:

                u_fun = interp1d(
                    self.data_processor.t_train, u_dash, axis=0, kind="cubic", fill_value="extrapolate"
                )

                def rhs(t, x):
                    # return self.models[self.model_name].predict(x[np.newaxis, :], u_fun(t))[0]
                    # uk = u_train_dash[k - 1, :]
                    # second param x should be 1d, first arg in concat should be 2d
                    # u_fun_train(t) should be 1d, secind arg in concat should be 2d
                    # output of rhs should be 1d
                    xk = x[np.newaxis, :]
                    x_shape = np.shape(xk)
                    uk = u_fun(t).reshape(1, -1)
                    X = np.concatenate((xk, uk), axis=1)  # .reshape(x_shape)
                    Xt = X
                    for _, _, transform in self.model.model._iter(with_final=False):
                        Xt = transform.transform(Xt)

                    return \
                    (safe_sparse_dot(Xt, self.model.model.steps[-1][1].optimizer.coef_.T, dense_output=True)
                     + self.model.model.steps[-1][1].optimizer.intercept_).reshape(x_shape)[0]

                # simulate the x states
                x_sim = (
                    (solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t)).y
                ).T

            else:

                x_sim = np.zeros((len(t), len(x0)))
                x_sim[0, :] = x0

                for k in range(1, len(t)):
                    xk = x_sim[k - 1, :]
                    uk = u_dash[k - 1, :]
                    Xt = np.concatenate((xk, uk))[np.newaxis, :]

                    for _, name, transform in self.model.model._iter(with_final=False):
                        Xt = transform.transform(Xt)

                    x_sim[k, :] = safe_sparse_dot(Xt,
                                                  self.model.model.steps[-1][1].optimizer.coef_.T,
                                                  dense_output=True) \
                                  + self.model.model.steps[-1][1].optimizer.intercept_
        else:
            x_sim = np.zeros((len(t), len(x0)))
            x_sim[0, :] = x0

            for k in range(len(t)):
                xk = x_sim[k, :]
                uk = u_dash[k, :]
                Xt = np.concatenate((xk, uk))[np.newaxis, :]

                for _, name, transform in self.model.model._iter(with_final=False):
                    Xt = transform.transform(Xt)

                x_sim[k, :] = safe_sparse_dot(Xt,
                                              self.model.model.steps[-1][1].optimizer.coef_.T,
                                              dense_output=True)

        return t, x_sim

    def simulate_cases(self, test_indices, **kwargs):
        cases = np.array([None for case_idx in range(self.data_processor.n_datasets)])
        if kwargs['parallel']:
            pool = Pool(mp.cpu_count())
            cases = np.array([None for case_idx in range(self.data_processor.n_datasets)])
            cases[test_indices] = pool.starmap(simulate_case, [(self.model_name, case_idx,
                      self.data_processor.training_idx, self.data_processor.testing_idx,
                      self.data_processor.x_train, self.data_processor.x_dot_train,
                      self.data_processor.u_train, self.data_processor.t_train,
                      self.data_processor.x_test, self.data_processor.x_dot_test,
                      self.data_processor.u_test, self.data_processor.t_test,
                      self.data_processor.x_saturation,
                      self.data_processor.ctrl_inpt_cols,
                      [item[-1].transform for item in self.model.model._iter(with_final=False)],
                      self.model.model.steps[-1][1].optimizer.coef_,
                      self.model.model.steps[-1][1].optimizer.intercept_,
                     kwargs['is_discrete'], kwargs['is_dynamical'])
                                         for case_idx in range(self.data_processor.n_datasets) if case_idx in test_indices])
        else:
            for case_idx in range(self.data_processor.n_datasets):
                if case_idx in test_indices:

                    # if self.data_processor.model_per_wind_speed:
                    #     if test_on_training_data:
                    #         model_name = f'{self.model_name}_{self.data_processor["vmean_train"][case_idx]}'
                    #     else:
                    #         model_name = f'{self.model_name}_{self.data_processor["vmean_test"][case_idx]}'
                    # else:
                    #     model_name = self.model_name

                    # yield self.simulate_case(case_idx, **kwargs)
                    cases[case_idx] = simulate_case(self.model_name, case_idx,
                      self.data_processor.training_idx, self.data_processor.testing_idx,
                      self.data_processor.x_train, self.data_processor.x_dot_train,
                      self.data_processor.u_train, self.data_processor.t_train,
                      self.data_processor.x_test, self.data_processor.x_dot_test,
                      self.data_processor.u_test, self.data_processor.t_test,
                                                    self.data_processor.x_saturation,
                      self.data_processor.ctrl_inpt_cols,
                      [item[-1].transform for item in self.model.model._iter(with_final=False)],
                      self.model.model.steps[-1][1].optimizer.coef_,
                      self.model.model.steps[-1][1].optimizer.intercept_,
                      kwargs['is_discrete'], kwargs['is_dynamical'])
                # else:
                #     yield None
        return cases

    def save_simulations(self, results_dir, sims_list):
        with open(os.path.join(results_dir, f'{self.model_name}_sim.txt'), 'wb') as fh:
            pickle.dump(sims_list, fh)

    def load_simulations(self, results_dir):

        with open(os.path.join(results_dir, f'{self.model_name}_sim.txt'), 'rb') as fh:
            sims_list = pickle.load(fh)

        return sims_list

def simulate_case(model_name, test_idx,
                  training_idx, testing_idx,
                  x_train, x_dot_train, u_train, t_train,
                  x_test, x_dot_test, u_test, t_test,
                  x_saturation,
                  ctrl_inpt_cols,
                  transform_iter, coeffs, intercept,
                  is_discrete=False, is_dynamical=True):
    
    print(f'\nSimulating {model_name} for case {test_idx}')

    # full_lib.library_ensemble = False

    if test_idx in training_idx:
        dataset_type_idx = np.argwhere(training_idx == test_idx).squeeze()
        x = x_train[model_name][dataset_type_idx]
        # x_dot = x_dot_train[model_name][dataset_type_idx]
        u = u_train[model_name][dataset_type_idx]
        t = t_train
    elif test_idx in testing_idx:
        dataset_type_idx = np.argwhere(testing_idx == test_idx).squeeze()
        x = x_test[model_name][dataset_type_idx]
        # x_dot = x_dot_test[model_name][dataset_type_idx]
        u = u_test[model_name][dataset_type_idx]
        t = t_test

    u_dash = u
    x0 = x[0, :]

    if is_dynamical:
        if not is_discrete:

            u_fun = interp1d(
                t_train, u_dash, axis=0, kind="cubic", fill_value="extrapolate"
            )

            # + self.model.model.steps[-1][1].optimizer.intercept_).reshape(x_shape)[0]

            # simulate the x states
            x_sim = (
                (solve_ivp(
                    lambda t, x: ct_rhs(t, x, u_fun, transform_iter, coeffs, intercept),
                           (t[0], t[-1]),
                           x0,
                           t_eval=t)).y
            ).T
            if x_saturation[model_name][0] is not None or x_saturation[model_name][0] is not None:
                x_sim = np.clip(x_sim, x_saturation[model_name][0], x_saturation[model_name][1])

        else:

            x_sim = np.zeros((len(t), len(x0)))
            x_sim[0, :] = x0
            for k in range(1, len(t)):
                xk = x_sim[k - 1, :]
                uk = u_dash[k - 1, :]
                Xt = np.concatenate((xk, uk))[np.newaxis, :]
                
                try:
                    for transform in transform_iter:
                        Xt = transform(Xt)
                except ValueError as e:
                    return t, x_sim

                x_sim[k, :] = safe_sparse_dot(Xt,
                                              coeffs.T,
                                              dense_output=True) + intercept

                if x_saturation[model_name][0] is not None or x_saturation[model_name][0] is not None:
                    x_sim[k, :] = np.clip(x_sim[k, :], x_saturation[model_name][0], x_saturation[model_name][1])
                    
                    
    else:
        x_sim = np.zeros((len(t), len(x0)))
        x_sim[0, :] = x0

        for k in range(len(t)):
            xk = x_sim[k, :]
            uk = u_dash[k, :]
            Xt = np.concatenate((xk, uk))[np.newaxis, :]

            for transform in transform_iter:
                Xt = transform(Xt)

            x_sim[k, :] = safe_sparse_dot(Xt,
                                          coeffs.T,
                                          dense_output=True) + intercept
            
            if x_saturation[model_name][0] is not None or x_saturation[model_name][0] is not None:
                x_sim[k, :] = np.clip(x_sim[k, :], x_saturation[model_name][0], x_saturation[model_name][1])

    return t, x_sim


def ct_rhs(t, x, u_fun, transform_iter, coeffs, intercept):
    # return self.models[model_name].predict(x[np.newaxis, :], u_fun(t))[0]
    # uk = u_train_dash[k - 1, :]
    # second param x should be 1d, first arg in concat should be 2d
    # u_fun_train(t) should be 1d, secind arg in concat should be 2d
    # output of rhs should be 1d
    xk = x[np.newaxis, :]
    x_shape = np.shape(xk)
    uk = u_fun(t).reshape(1, -1)
    X = np.concatenate((xk, uk), axis=1)  # .reshape(x_shape)
    Xt = X
    for transform in transform_iter:
        Xt = transform(Xt)

    return \
        (safe_sparse_dot(Xt, coeffs.T, dense_output=True)
         + intercept).reshape(x_shape)[0]