import pickle
from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame as df
import numpy as np
from pmdarima.arima import auto_arima
from tqdm import tqdm
# This is because auto_arima warns about differencing.
import warnings
warnings.filterwarnings("ignore")


def system_A(T_1, C_1, T_2, C_2, W, P, k, N, learning_function, csv_directory):
    """
    INPUTS:
    T_1. The time index for the first regime. The default is {1, ..., 29, 30}
    C_1. The coefficients of the DGP for the first regime. This will be a list containing [y_0, c, \phi].
    T_2. The time index for the second regime. The default is {31, ..., 62, 63}
    C_2. The coefficients of the DGP for the second regime. This will be a list containing [d, \phi_1, \phi_2].
    W. Window sizes. Default set is {20, 40}.
    P. AR orders. Default set is {1, 2, 3}.
    k. Forecast horizon. We fix k = 3.
    N. The number of iterations for the simulation.
    p. The p specification for adaptive learning.
    learning_function. Includes the parameters [v, p, \lambda, g] used in the learning process.
    OUTPUTS:
    MSE for each fixed model and both EN learning and Ensemble Learning.
    Model Choices for all model selection techniques.
    """

    # Logistics

    # Regime 1
    y_0 = C_1[0]
    c = C_1[1]
    phi = C_1[2]

    # Regime 2
    d = C_2[0]
    phi_1 = C_2[1]
    phi_2 = C_2[2]

    # Adaptive Learning Logistics

    v, p_norm, Lambda, g = learning_function

    Lambda_vector = [Lambda**(v-i) for i in range(0, v)]
    v0 = 7
    v1 = v - v0
    Lambda_vector_Ensemble = [Lambda**(v1-i) for i in range(0, v1)]

    # Other logistics
    MSE_N = create_H_tilde_dict(P, W)
    H_tilde = MSE_N.keys()
    model_selections = create_H_tilde_dict(P, W)
    forecast_dict_N = create_H_tilde_dict(P, W)

    # Step 1:
    for s in tqdm(N, leave=True, position=0):

        # Step 1 (a): Set the seed.
        np.random.seed(s)

        # Step 1 (b): Declare y_0
        simulated_data = [y_0]

        # Step 1 (c): Simulate the first regime.
        for t in T_1:

            # Step 1 (c)(i): Generate and save y_t.
            epsilon_t = np.random.normal(loc=0, scale=1)

            y_t = c + phi*simulated_data[t-1] + epsilon_t

            simulated_data.append(y_t)

        # Step 1 (d): Simulate the second regime.
        for t in T_2:

            epsilon_t = np.random.normal(loc=0, scale=1)

            # Step 1 (d)(i): Generate and save y_t.
            y_t = d + phi_1*simulated_data[t-1] + \
                phi_2*simulated_data[t-2] + epsilon_t

            simulated_data.append(y_t)

        # Step 1 (e): Housekeeping
        T_train = np.arange(57 - v - 2, 61)

        # Logistics
        forecast_error_df = create_model_df(T_train, H_tilde)
        p_norm_df = create_model_df(T_train, H_tilde)
        denoised_p_norm_df = create_model_df(T_train, H_tilde)
        MC_p_norm_df = create_model_df(T_train, H_tilde)
        forecasts_t3 = create_model_df(T_train, H_tilde)
        forecasts_t2 = create_model_df(T_train, H_tilde)
        forecasts_t1 = create_model_df(T_train, H_tilde)
        matrix = np.matrix([[1, 0, 0], [-1, 1, 0], [0, -1, 1]])

        # Step 1 (f): Collecting scalar forecasting errors.
        for t in T_train:

            if t > T_train[2]:

                # Collect the p-norm for the model g
                validation_vector = [simulated_data[t]]*k

                # Step 1 (g)(i)(A): Obtain the vector of forecasts.
                forecasts = [
                    float(forecasts_t1.loc[t-1, g]),
                    float(forecasts_t2.loc[t-2, g]),
                    float(forecasts_t3.loc[t-3, g]),
                ]

                error_g = np.subtract(forecasts, validation_vector)

                p_norm_g = np.linalg.norm(
                    x=error_g, ord=p_norm)**p_norm

            for AR in P:
                for w in W:

                    # Step 1 (f)(i)(A): train the model
                    ### Single - Value Forecast Errors ###
                    model = ARIMA(simulated_data[(t - w + 1): (t + 1)], order=(
                        AR, 0, 0)).fit(method="yule_walker")

                    model_name = naming_function(AR, w)

                    # Step 1 (f)(i)(B): produce and save the single forecast
                    forecasts = model.forecast(k)

                    single_forecast = forecasts[-1]

                    forecasts_t3.loc[t, model_name] = single_forecast
                    forecasts_t2.loc[t, model_name] = forecasts[1]
                    forecasts_t1.loc[t, model_name] = forecasts[0]

                    forecast_dict_N[model_name][s] = single_forecast

                    validation_data = simulated_data[t+k]

                    forecast_error = (validation_data - single_forecast)

                    forecast_error_df.loc[t+k,
                                          model_name] = abs(forecast_error)**p_norm

                    if t > T_train[2]:

                        # Step 1 (g): Collecting multi-value forecasting errors.
                        validation_vector = [simulated_data[t]]*k

                        # Step 1 (g)(i)(A): Obtain the vector of forecasts.
                        forecasts = [
                            float(forecasts_t1.loc[t-1, model_name]),
                            float(forecasts_t2.loc[t-2, model_name]),
                            float(forecasts_t3.loc[t-3, model_name]),
                        ]

                        # Step 1 (g)(i)(B): Collecting multi-value forecasting errors.
                        vector_errors = np.subtract(
                            forecasts, validation_vector)

                        # Step 1 (g)(i)(C): Collecting denoised multi-value forecasting errors.
                        denoised_error = np.asarray(denoise_forecast_error(
                            vector_errors, matrix)).reshape(-1)

                        p_norm_t = np.linalg.norm(
                            x=vector_errors, ord=p_norm)**p_norm

                        p_norm_df.loc[t, model_name] = p_norm_t

                        denoised_p_norm = np.linalg.norm(
                            x=denoised_error, ord=p_norm)**p_norm

                        denoised_p_norm_df.loc[t, model_name] = denoised_p_norm

                        MC_p_norm_df.loc[t, model_name] = p_norm_t/p_norm_g

        t = 60
        for p in P:
            for w in W:

                model_name = naming_function(p, w)

                # Step 1 (h)(i)(A): Obtain the forecast for the fixed model groups
                forecast = float(forecasts_t3.loc[t, model_name])

                # Step 1 (h)(i)(B): Save the MSE for the fixed model groups
                MSE = (simulated_data[-1] - forecast)**2

                MSE_N[model_name][s] = MSE

        # Step 1 (h)(ii): Declare \tilde{T}.
        T_tilde = np.arange(t - v + 1, t + 1)

        # Step 1 (h)(iii): Perform Adaptive Learning at t = 60.

        # EN
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

        forecast_dict_N["EN"][s] = y_star
        forecast_dict_N['EN Multi-Valued'][s] = y_star_multi
        forecast_dict_N['EN Multi-Valued Denoised'][s] = y_star_multi_denoised
        forecast_dict_N['EN MC'][s] = y_star_MC
        forecast_dict_N['Ensemble'][s] = y_star_ensemble
        forecast_dict_N['Ensemble Multi-Valued'][s] = y_star_ensemble_multi
        forecast_dict_N['Ensemble Multi-Valued Denoised'][s] = y_star_ensemble_denoised
        forecast_dict_N['Ensemble MC'][s] = y_star_ensemble_MC

        # EN: single-valued.
        MSE_EN = (simulated_data[-1] - y_star)**2
        MSE_N['EN'][s] = MSE_EN

        # EN: multi-valued.
        MSE_EN_multi = (simulated_data[-1] - y_star_multi)**2
        MSE_N['EN Multi-Valued'][s] = MSE_EN_multi

        # EN: multi-valued denoised.
        MSE_EN_multi_denoised = (simulated_data[-1] - y_star_multi_denoised)**2
        MSE_N['EN Multi-Valued Denoised'][s] = MSE_EN_multi_denoised

        # EN: MC
        MSE_EN_MC = (simulated_data[-1] - y_star_MC)**2
        MSE_N['EN MC'][s] = MSE_EN_MC

        # Ensemble: single-valued
        MSE_Ensemble = (simulated_data[-1] - y_star_ensemble)**2
        MSE_N['Ensemble'][s] = MSE_Ensemble

        # Ensemble: multi-valued
        MSE_Ensemble_multi = (simulated_data[-1] - y_star_ensemble_multi)**2
        MSE_N['Ensemble Multi-Valued'][s] = MSE_Ensemble_multi

        # Ensemble: multi-valued denoised
        MSE_Ensemble_denoised = (
            simulated_data[-1] - y_star_ensemble_denoised)**2
        MSE_N['Ensemble Multi-Valued Denoised'][s] = MSE_Ensemble_denoised

        # Ensemble: MC
        MSE_Ensemble_MC = (
            simulated_data[-1] - y_star_ensemble_MC)**2
        MSE_N['Ensemble MC'][s] = MSE_Ensemble_MC

        # AIC and BIC
        for w in W:
            for criterion in ['bic', 'aic']:

                model = auto_arima(y=simulated_data[(
                    t - w + 1): (t + 1)], max_p=3, start_d=0, max_d=1,  max_q=1, start_q=0, information_criterion=criterion)

                p = int(str(model)[7])
                d = int(str(model)[9])
                q = 0

                model_selections[f"{criterion.upper()} w{w}"][s] = "(" + \
                    f"{p}, {d}, {q}" + ")"

                model = ARIMA(simulated_data[(t - w + 1): (t + 1)], order=(
                    p, d, q)).fit(method="yule_walker")

                forecast = model.forecast(k)[-1]

                MSE = (simulated_data[-1] - forecast)**2

                MSE_N[f"{criterion.upper()} w{w}"][s] = MSE

                forecast_dict_N[f"{criterion.upper()} w{w}"][s] = forecast

    # Step 2: return the output
    with open(csv_directory + f'MSE_N_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(MSE_N, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(csv_directory + f'model_selections_N_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(model_selections, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return MSE_N, model_selections, forecast_dict_N
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

### Helper Functions ###


def create_H_tilde_dict(P, W):
    H_tilde_dict = {}
    for p in P:
        for w in W:
            H_tilde_dict.update({naming_function(p, w): {}})
    H_tilde_dict.update({"EN": {}})
    H_tilde_dict.update({"Ensemble": {}})
    H_tilde_dict.update({'EN Multi-Valued': {}})
    H_tilde_dict.update({'Ensemble Multi-Valued': {}})
    H_tilde_dict.update({'Ensemble Multi-Valued Denoised': {}})
    H_tilde_dict.update({'EN Multi-Valued Denoised': {}})
    H_tilde_dict.update({'EN MC': {}})
    H_tilde_dict.update({'Ensemble MC': {}})
    for w in W:
        H_tilde_dict.update({f"AIC w{w}": {}})
        H_tilde_dict.update({f"BIC w{w}": {}})
    return H_tilde_dict


def create_value_dict(functional_sets):
    value_dict = {}
    for model in functional_sets:
        value_dict.update({model: 0})
    return value_dict


def create_model_df(T_train, functional_sets):
    headers = []
    for functional_set in functional_sets:
        if "en" not in functional_set.lower() and 'IC' not in functional_set:
            headers.append(functional_set)
    forecast_df = df(columns=[headers])
    forecast_df['time_index'] = T_train
    forecast_df.set_index(T_train, inplace=True)
    del forecast_df['time_index']
    return forecast_df


def naming_function(p, w):
    return f"(AR{p}, w{w})"
