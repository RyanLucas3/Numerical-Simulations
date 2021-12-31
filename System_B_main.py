import numpy as np
from itertools import product
from pandas import DataFrame as df
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
import pandas as pd
from pmdarima.arima import auto_arima
import warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")


class SimulationModelSpecs:
    def __init__(self, ar_orders, window_sizes, desired_model_groups):
        self.ar_orders = ar_orders
        self.window_sizes = window_sizes
        self.desired_model_groups = desired_model_groups

    def create_functional_sets(self):
        output = []

        if "MG1" in self.desired_model_groups:
            # MG1 = Model Group 1: AR models.

            self.MG1_model_specs = self.get_all_possible_combinations(
                model_group=['1'],
                MG_ar_orders=self.ar_orders,
                MG_window_sizes=self.window_sizes)
            output.append(self.MG1_model_specs)

        if "MG2V" in self.desired_model_groups:
            # MG3V: VAR(p) models on VIX and Yield short- and long-run rate pairs.

            self.MG2V_model_specs = self.get_all_possible_combinations(
                model_group=['2V'],
                MG_ar_orders=self.ar_orders,
                MG_window_sizes=self.window_sizes,
                MG_i=['x_1'])

            output.append(self.MG2V_model_specs)

        # Returning the functional sets to be deployed.
        return output

    def screen_models(self, models):
        screened_models = []
        for model in models:
            if model[0] == '2V':
                if 2*(2 + 4*model[1]) < model[2]:
                    screened_models.append(model)
            else:
                screened_models.append(model)
        return screened_models

    def get_all_possible_combinations(self, model_group, MG_ar_orders, MG_window_sizes, MG_i=[None]):
        models = list(product(model_group, MG_ar_orders,
                      MG_window_sizes, MG_i))
        return self.screen_models(models)


def system_B(functional_sets,
             m,
             T_1,
             T_2,
             Y_0,
             A,
             B,
             C,
             Pi_1,
             Pi_2,
             N,
             k,
             W,
             learning_function,
             csv_directory=None):
    """


   INPUTS:

   m. The number of variables to simulate.

   T_1. The time index for the first regime. The default is {2, ..., 29, 60}

   T_2. The time index for the second regime. The default is {61, ..., 122, 123}

   ---- SEE PSEUDO ALGO SECTION 1.4. FOR ALL DETAILS ON STARTING VALUES AND COEFFICIENTS ----

   Y_0. The initial values for the VAR system. See section 1.4 of the pseudo algo for details.

   A. The m x 1 vector of interecept parametters for the first regime [VAR(1)],
   where m is the number of system variables to simulate.

   B.  The m x m matrix of endogenous variable parameters for the first regime.

   C. The m x 1 vector of interecept parametters for the first regime,

   Pi_1. The m x m first order matrix of endogenous variable parameters for the second regime [VAR(2)].

   Pi_2. The m x m second order matrix of endogenous variable parameters for the second regime.

   ---- ------------------------------------------------------------------------------- ----

   N. The number of iterations for the simulation.

   k. Forecast horizon. We fix k = 3.

   W. Window sizes. Default set is {20, 40}.

   Functional sets. The total set of models to be trained.

   P. AR orders. Default set is {1, 2, 3}.

   OUTPUTS:

   MSE Values for each model across N simulations.

   Model selection information for Adaptive Learning and information criterion models across N simulations.

   """

    # Adaptive Learning Logistics
    v, p_norm, Lambda, g = learning_function
    Lambda_vector = [Lambda**(v-i) for i in range(0, v)]
    v0 = 7
    v1 = v - v0
    Lambda_vector_Ensemble = [Lambda**(v1-i) for i in range(0, v1)]
    matrix = np.matrix([[1, 0, 0], [-1, 1, 0], [0, -1, 1]])

    # Other Logistics
    MSE_N = create_H_tilde_dict(functional_sets)
    H_tilde = MSE_N.keys()
    model_selections = create_H_tilde_dict(functional_sets)

    for s in tqdm(N, position=0, leave=True):

        # Step 1: Set the seed.
        np.random.seed(s)

        # Step 2: Declare Y_0 and Y_1, which are assumed to be the same here.
        simulated_data = df({'y': [Y_0[0]],
                             'x_1': [Y_0[1]]})

        # Step 3: Generate the time series for the first regime.
        for t in T_1:

            epsilon_t = np.random.normal(0, 1, m)

            Y_t = A + \
                np.matmul(B, np.array(simulated_data.loc[t-1])) + epsilon_t

            Y_t = df(Y_t, columns=['y', 'x_1'])

            simulated_data = pd.concat(
                [simulated_data, Y_t], ignore_index=True, axis=0)

        # Step 4: Generate the time series for the second regime.
        for t in T_2:

            epsilon_t = np.random.normal(0, 1, m)

            Y_t = C + np.matmul(Pi_1, simulated_data.loc[t-1]) + np.matmul(
                Pi_2, simulated_data.loc[t-2]) + epsilon_t

            Y_t = df([Y_t], columns=['y', 'x_1'])

            simulated_data = pd.concat(
                [simulated_data, Y_t], ignore_index=True, axis=0)

        # Logistics
        T_train = np.arange(80 - v - 2, 81)
        forecast_error_df = create_model_df(T_train, H_tilde)
        p_norm_df = create_model_df(T_train, H_tilde)
        denoised_p_norm_df = create_model_df(T_train, H_tilde)
        MC_p_norm_df = create_model_df(T_train, H_tilde)
        forecasts_t3 = create_model_df(T_train, H_tilde)
        forecasts_t2 = create_model_df(T_train, H_tilde)
        forecasts_t1 = create_model_df(T_train, H_tilde)

        # Step 5: Training, forecasting and collecting of necessary loss metrics.
        for t in T_train:

            if t > T_train[k-1]:

                # Step 5 (a)(i): For the designated model g.
                validation_vector = [simulated_data.loc[t, 'y']]*k

                # Step 5 (a)(ii): Obtain the vector of forecasts.
                forecasts = [
                    float(forecasts_t1.loc[t-1, g]),
                    float(forecasts_t2.loc[t-2, g]),
                    float(forecasts_t3.loc[t-3, g]),
                ]

                error_g = np.subtract(forecasts, validation_vector)

                # Step 5 (a)(iii): Obtain the p-norm for the model g.
                p_norm_g = np.linalg.norm(
                    x=error_g, ord=p_norm)**p_norm

            # Step 5 (b): Training
            for functional_set in functional_sets:

                for model_group, ar_order, window_size, i in functional_set:

                    model_name = naming_function(
                        model_group, ar_order, window_size)

                    # Step 5 (b)(i): Train the model h according to the model group.

                    if model_group == '2V':

                        data = np.column_stack([simulated_data['y'],
                                                simulated_data[i]])

                        forecasts = train_and_forecast_VAR(data,
                                                           t,
                                                           window_size,
                                                           ar_order,
                                                           k)

                    # Step 5 (b)(ii): Produce and save the vector of forecasts [y_t+3|t, y_t+2|t, y_t+1|t]
                    forecasts_t3.loc[t, model_name] = forecasts[2]
                    forecasts_t2.loc[t, model_name] = forecasts[1]
                    forecasts_t1.loc[t, model_name] = forecasts[0]

                    validation_data = simulated_data.loc[t+k, 'y']

                    # Step 5 (b)(iii): Obtain and save the scalar forecast error.
                    forecast_error = (validation_data -
                                      forecasts_t3.loc[t, model_name])

                    forecast_error_df.loc[t+k,
                                          model_name] = abs(forecast_error)**p_norm

                    # Step 5 (b)(iv)
                    if t > T_train[k-1]:

                        validation_vector = [simulated_data.loc[t, 'y']]*k

                        # Step 5 (b)(iv)(A): Obtain the vector of forecasts [y_t|t-1, y_t|t-2, y_t|t-3].
                        forecasts = [
                            float(forecasts_t1.loc[t-1, model_name]),
                            float(forecasts_t2.loc[t-2, model_name]),
                            float(forecasts_t3.loc[t-3, model_name]),
                        ]

                        # Step 5 (b)(iv)(B): Obtain the vector multi-valued forecast error.
                        vector_errors = np.subtract(
                            forecasts, validation_vector)

                        p_norm_t = np.linalg.norm(
                            x=vector_errors, ord=p_norm)**p_norm

                        p_norm_df.loc[t, model_name] = p_norm_t

                        # Step 5 (b)(iv)(C): Obtain the vector multi-valued denoised forecast error.
                        denoised_error = np.asarray(denoise_forecast_error(
                            vector_errors, matrix)).reshape(-1)

                        denoised_p_norm = np.linalg.norm(
                            x=denoised_error, ord=p_norm)**p_norm

                        denoised_p_norm_df.loc[t,
                                               model_name] = denoised_p_norm

                        # Step 5 (b)(iv)(C): Obtain the comparison forecast error.
                        MC_p_norm_df.loc[t, model_name] = p_norm_t/p_norm_g

        # Step 6: Collect MSE at t = 120 for Fixed models and Adaptive Learning models.
        t = 80
        for functional_set in functional_sets:

            for model_group, ar_order, window_size, i in functional_set:

                model_name = naming_function(
                    model_group, ar_order, window_size)

                # Step 1 (h)(i)(A): Obtain the forecast for the fixed model groups
                forecast = float(forecasts_t3.loc[t, model_name])

                # Step 1 (h)(i)(B): Save the MSE for the fixed model groups
                MSE = (simulated_data.loc[83, 'y'] - forecast)**2

                MSE_N[model_name][s] = MSE

        # Step 1 (h)(ii): Declare \tilde{T}.
        T_tilde = np.arange(t - v + 1, t + 1)

        # # Step 1 (h)(iii): Perform Adaptive Learning at t = 120.

        # # EN
        y_star, h_star = EN_learning(t,
                                     forecast_error_df,
                                     T_tilde,
                                     forecasts_t3,
                                     Lambda_vector)

        # EN Multi
        y_star_multi, h_star_multi = EN_learning(t,
                                                 p_norm_df,
                                                 T_tilde,
                                                 forecasts_t3,
                                                 Lambda_vector)

        # EN Denoised
        y_star_multi_denoised, h_star_multi_denoised = EN_learning(t,
                                                                   denoised_p_norm_df,
                                                                   T_tilde,
                                                                   forecasts_t3,
                                                                   Lambda_vector)

        # MC EN.
        y_star_MC, h_star_MC = EN_learning(t,
                                           MC_p_norm_df,
                                           T_tilde,
                                           forecasts_t3,
                                           Lambda_vector)

        # Ensemble
        y_star_ensemble, h_star_ensemble = Ensemble(t,
                                                    v0,
                                                    v1,
                                                    Lambda_vector_Ensemble,
                                                    forecasts_t3.columns,
                                                    forecasts_t3,
                                                    forecast_error_df)
        # Ensemble Multi
        y_star_ensemble_multi, h_star_ensemble_multi = Ensemble(t,
                                                                v0,
                                                                v1,
                                                                Lambda_vector_Ensemble,
                                                                forecasts_t3.columns,
                                                                forecasts_t3,
                                                                p_norm_df)
        # Ensemble Denoised
        y_star_ensemble_denoised, h_star_ensemble_denoised = Ensemble(t,
                                                                      v0,
                                                                      v1,
                                                                      Lambda_vector_Ensemble,
                                                                      forecasts_t3.columns,
                                                                      forecasts_t3,
                                                                      denoised_p_norm_df)

        # Ensemble MC
        y_star_ensemble_MC, h_star_ensemble_MC = Ensemble(t,
                                                          v0,
                                                          v1,
                                                          Lambda_vector_Ensemble,
                                                          forecasts_t3.columns,
                                                          forecasts_t3,
                                                          MC_p_norm_df)

        model_selections["EN"][s] = h_star
        model_selections['EN Multi-Valued'][s] = h_star_multi
        model_selections['EN Multi-Valued Denoised'][s] = h_star_multi_denoised
        model_selections['EN MC'][s] = h_star_MC
        model_selections['Ensemble'][s] = h_star_ensemble
        model_selections['Ensemble Multi-Valued'][s] = h_star_ensemble_multi
        model_selections['Ensemble Multi-Valued Denoised'][s] = h_star_ensemble_denoised
        model_selections['Ensemble MC'][s] = h_star_ensemble_MC

        # EN: single-valued.
        MSE_EN = (simulated_data.loc[83, 'y'] - y_star)**2
        MSE_N['EN'][s] = MSE_EN

        # EN: multi-valued.
        MSE_EN_multi = (simulated_data.loc[83, 'y'] - y_star_multi)**2
        MSE_N['EN Multi-Valued'][s] = MSE_EN_multi

        # EN: multi-valued denoised.
        MSE_EN_multi_denoised = (
            simulated_data.loc[83, 'y'] - y_star_multi_denoised)**2
        MSE_N['EN Multi-Valued Denoised'][s] = MSE_EN_multi_denoised

        # EN: MC
        MSE_EN_MC = (simulated_data.loc[83, 'y'] - y_star_MC)**2
        MSE_N['EN MC'][s] = MSE_EN_MC

        # Ensemble: single-valued
        MSE_Ensemble = (simulated_data.loc[83, 'y'] - y_star_ensemble)**2
        MSE_N['Ensemble'][s] = MSE_Ensemble

        # Ensemble: multi-valued
        MSE_Ensemble = (
            simulated_data.loc[83, 'y'] - y_star_ensemble_multi)**2
        MSE_N['Ensemble Multi-Valued'][s] = MSE_Ensemble

        # Ensemble: multi-valued denoised
        MSE_Ensemble_denoised = (
            simulated_data.loc[83, 'y'] - y_star_ensemble_denoised)**2
        MSE_N['Ensemble Multi-Valued Denoised'][s] = MSE_Ensemble_denoised

        # Ensemble: MC
        MSE_Ensemble_MC = (
            simulated_data.loc[83, 'y'] - y_star_ensemble_MC)**2
        MSE_N['Ensemble MC'][s] = MSE_Ensemble_MC

        # AIC and BIC
        for w in W:
            for criterion in ['bic', 'aic']:

                # windowed_data = simulated_data.loc[(
                #     t - w): (t), 'y']

                # # # Univariate AIC.
                # model = auto_arima(y=windowed_data, max_p=3, start_d=0, max_d=1,
                #                    max_q=1, start_q=0, information_criterion=criterion)

                # p = int(str(model)[7])
                # d = int(str(model)[9])
                # q = 0

                # model_selections[f"{criterion.upper()} AR w{w}"][s] = "(" + \
                #     f"{p}, {d}, {q}" + ")"

                # model = ARIMA(simulated_data.loc[(t - w): (t), 'y'], order=(
                #     p, d, q)).fit(method="yule_walker")

                # forecast = list(model.forecast(k))[-1]

                # MSE = (simulated_data.loc[83, 'y'] - forecast)**2

                # MSE_N[f"{criterion.upper()} AR w{w}"][s] = MSE

                # Bivariate (VAR) AIC.

                VAR_data = simulated_data[(t - w + 1): (t + 1)]

                VAR_model = VAR(VAR_data).fit(maxlags=3, ic=criterion)

                VAR_p = VAR_model.k_ar

                if VAR_p == 0:

                    VAR_forecast = float(VAR_model.params['y'])

                else:

                    VAR_forecast = VAR_model.forecast(
                        y=np.array(simulated_data[(t - (VAR_p) + 1): (t + 1)]), steps=k)[:, 0][-1]

                MSE_VAR = (simulated_data.loc[83, 'y'] - VAR_forecast)**2

                MSE_N[f'{criterion.upper()} VAR w{w}'][s] = MSE_VAR

                model_selections[f"{criterion.upper()} VAR w{w}"][s] = "(" + \
                    f"{VAR_p}" + ")"

    with open(csv_directory + f'MSE_B2_{N[0]}_{N[-1]}_newDGP.pickle', 'wb') as handle:
        pickle.dump(MSE_N, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(csv_directory + f'model_selections_B2_{N[0]}_{N[-1]}_newDGP.pickle', 'wb') as handle:
        pickle.dump(model_selections, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    return MSE_N, model_selections


def train_and_forecast_MG1(data, t, w, ar_order, k):

    windowed_data = np.array(data[(t - w + 1): (t + 1)])

    model = ARIMA(windowed_data, order=(
        ar_order, 0, 0)).fit(method="yule_walker")

    forecasts = model.forecast(k)

    return forecasts


def train_and_forecast_VAR(data, t, w, ar_order, k):

    windowed_data = data[(t - w + 1): (t + 1)]

    model = VAR(windowed_data).fit(ar_order)

    forecasts = model.forecast(
        y=data[(t - (ar_order) + 1): (t + 1), :], steps=k)[:, 0]

    return forecasts


def create_H_tilde_dict(H):
    H_tilde_dict = {}
    for functional_set in H:
        for model_group, ar_order, window_size, i in functional_set:
            H_tilde_dict.update(
                {naming_function(model_group, ar_order, window_size): {}})
            # H_tilde_dict.update({f"AIC AR w{window_size}": {}})
            # H_tilde_dict.update({f"BIC AR w{window_size}": {}})
            H_tilde_dict.update({f"AIC VAR w{window_size}": {}})
            H_tilde_dict.update({f"BIC VAR w{window_size}": {}})

    H_tilde_dict.update({"EN": {}})
    H_tilde_dict.update({"Ensemble": {}})
    H_tilde_dict.update({'EN Multi-Valued': {}})
    H_tilde_dict.update({"EN MC": {}})
    H_tilde_dict.update({'Ensemble Multi-Valued': {}})
    H_tilde_dict.update({'Ensemble Multi-Valued Denoised': {}})
    H_tilde_dict.update({'EN Multi-Valued Denoised': {}})
    H_tilde_dict.update({'Ensemble MC': {}})
    return H_tilde_dict


def naming_function(model_group, ar_order, window_size):
    if model_group == "2V":
        return f"(VAR{ar_order}, w{window_size})"
    elif model_group == '1':
        return f"(AR{ar_order}, w{window_size})"


def create_model_df(T_train, functional_sets):
    headers = []
    functional_sets = list(filter(None, functional_sets))
    for functional_set in functional_sets:
        if "en" not in functional_set.lower() and 'IC' not in functional_set:
            headers.append(functional_set)
    forecast_df = df(columns=[headers])
    forecast_df['time_index'] = T_train
    forecast_df.set_index(T_train, inplace=True)
    del forecast_df['time_index']
    return forecast_df

    ### Adaptive Learning Functions ###


def EN_learning(t, forecast_error_df, T_tilde, forecast_df, Lambda_vector):

    errors_T_tilde = forecast_error_df.loc[T_tilde].apply(
        lambda x: exponential_learning(x, Lambda_vector))

    h_star = errors_T_tilde.idxmin(axis=1)

    y_star = forecast_df.loc[t, h_star]

    return y_star, h_star


def denoise_forecast_error(error_vector, matrix):
    return np.matmul(matrix, error_vector)


def exponential_learning(forecast_errors, lambda_vector):
    return np.dot(forecast_errors, lambda_vector)


def create_value_dict(functional_sets):
    value_dict = {}
    for model in functional_sets:
        value_dict.update({model: 0})
    return value_dict


def Ensemble(t, v0, v1, Lambda_vector, functional_sets, forecast_df, forecast_error_df):

    T_0 = np.arange(t - v0 + 1, t+1)

    minimiser_count = create_value_dict(forecast_df.columns)

    for r in T_0:

        T_1 = np.arange(r - v1 + 1, r+1)

        errors_T_1 = forecast_error_df.loc[T_1].apply(
            lambda x: exponential_learning(x, Lambda_vector))

        h_star_r = errors_T_1.idxmin(axis=1)

        minimiser_count[h_star_r] += 1

    ensemble_weights = {model: count/len(T_0)
                        for model, count in minimiser_count.items()}

    forecasts_candidates = [float(forecast_df.loc[t, model])
                            for model in functional_sets]

    ensembled_forecast = np.dot(
        list(ensemble_weights.values()), forecasts_candidates)

    return ensembled_forecast, ensemble_weights
