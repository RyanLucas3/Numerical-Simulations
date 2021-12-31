import io
import pstats
import cProfile
import pickle
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from pandas import DataFrame as df
from tqdm import tqdm
# This is because auto_arima warns about differencing.
import warnings
warnings.filterwarnings("ignore")

def system_C(t_simu,
             T,
             C_1,
             C_2,
             W,
             P,
             k,
             N,
             learning_function,
             csv_directory):

    # DGP Logistics
    S_0, gamma_1, sigma_1 = C_1
    gamma_2, sigma_2 = C_2
    T_max = max(T)

    # Adaptive Learning Logistics
    v, p_norm, Lambda, g = learning_function
    Lambda_vector = [Lambda**(v-i) for i in range(0, v)]
    v0 = 7
    v1 = v - v0
    Lambda_vector_Ensemble = [Lambda**(v1-i) for i in range(0, v1)]

    # Other logistics
    AMSE_N = create_H_tilde_dict(P, W)
    Sharpe_Ratios_N = create_H_tilde_dict(P, W)
    Annualised_Returns_N = create_H_tilde_dict(P, W)
    Max_DD_N = create_H_tilde_dict(P, W)
    H_tilde = AMSE_N.keys()
    model_selections_N = {}
    perc_correct_N = create_H_tilde_dict(P, W)
    time_series = {}
    t_simu_N = {}
    daily_profit_dict_N = {}

    for i in tqdm(N, position=0, leave=True):

        # Step 1: Set the seed.
        np.random.seed(i)

        # Step 2: Declare S_0.
        simulated_S = [S_0]
        simulated_y = [0, 0, 0]

        # Step 3: Generate t^{simu}, the index of the regime switch.
        # t^{simu} ~ Unif([50, 70])
        t_simu = np.random.choice([50, 70])

        # Step 4: Declare T^1 and T^2 based on the index of the regime switch.
        T_1 = np.arange(1, t_simu+1)
        T_2 = np.arange(t_simu + 1, T_max + 1)

        # Step 5: Generating the first regime.
        for t in T_1:

            epsilon_t = epsilon_t = np.random.normal(loc=0, scale=1)

            # Step 5 (a): Generate the stock price based on the first SDE.
            S_t = simulated_S[t-1] * np.exp((gamma_1 - (1/2)
                                            * sigma_1**2) * (1) + sigma_1 * epsilon_t)

            simulated_S.append(S_t)

            # Step 5 (b): Generate the simulated return based on the stock price.
            if t >= 3:
                y_t = np.log(S_t) - np.log(simulated_S[t-k])
                simulated_y.append(y_t)

        # Step 6: Generating the second regime.
        for t in T_2:

            epsilon_t = np.random.normal(loc=0, scale=1)

            # Step 6 (a): Generate the stock price based on the second SDE.
            S_t = simulated_S[t-1] * np.exp((gamma_2 - (1/2)
                                            * sigma_2**2) * (1) + sigma_2 * epsilon_t)

            simulated_S.append(S_t)

            # Step 6 (b): Generate the simulated return based on the stock price.
            y_t = np.log(S_t) - np.log(simulated_S[t-k])

            simulated_y.append(y_t)

        time_series[i] = simulated_y
        t_simu_N[i] = t_simu

        # Step 7: Define T^{train}.
        T_train = np.arange(max(P) + max(W) + 1, 111)

        # Logistics
        forecast_error_df = create_model_df(T_train, H_tilde)
        p_norm_df = create_model_df(T_train, H_tilde)
        MC_p_norm_df = create_model_df(T_train, H_tilde)
        forecasts_t3 = create_model_df(T_train, H_tilde)
        forecasts_t2 = create_model_df(T_train, H_tilde)
        forecasts_t1 = create_model_df(T_train, H_tilde)
        MSE_t = create_H_tilde_dict(P, W)
        model_selections_t = create_H_tilde_dict(P, W)
        dir_correct_t = create_all_value_dict(P, W)
        # Step 8: Collecting forecasting errors and necessary adaptive learning loss metrics.
        for t in T_train:

            # Step 8 (a): For the designated model g.
            if t > T_train[2]:

                # Step 8 (a)(i): Collect the p-norm.
                validation_vector = [simulated_y[t]]*k

                # Step 8 (a)(ii): Obtain the error vector.
                forecasts = [
                    float(forecasts_t1.loc[t-1, g]),
                    float(forecasts_t2.loc[t-2, g]),
                    float(forecasts_t3.loc[t-3, g]),
                ]

                error_g = np.subtract(forecasts, validation_vector)

                # Step 8 (a)(iii): Obtain the error vector.
                p_norm_g = np.linalg.norm(
                    x=error_g, ord=p_norm)**p_norm

            # Step 8 (b): Train the fixed models and obtain the forecasts and forecasting errors.
            for AR in P:
                for w in W:

                    # Step 8 (b)(A): Train the fixed model.
                    model = ARIMA(simulated_y[(t - w + 1): (t + 1)], order=(
                        AR, 0, 0)).fit(method="yule_walker")

                    model_name = naming_function(AR, w)

                    # Step 8 (b)(B): produce and save the forcasts [y_{t+3|t}, y_{t+2|t}, y_{t+1|t}]
                    forecasts = model.forecast(k)

                    forecasts_t3.loc[t, model_name] = forecasts[2]
                    forecasts_t2.loc[t, model_name] = forecasts[1]
                    forecasts_t1.loc[t, model_name] = forecasts[0]

                    validation_data = simulated_y[t+k]

                    # Step 8 (b)(C): Obtain and save the scalar forecast error.
                    # forecasts[2] = y_{t+3|t}
                    forecast_error = (validation_data - forecasts[2])
                    forecast_error_df.loc[t+k,
                                          model_name] = abs(forecast_error)**p_norm

                    # Step 8 (b)(D): If t >= max(P) + max(W) + 4, collect vector forecast erros.
                    if t > T_train[2]:

                        # Validation vector of length k.
                        validation_vector = [simulated_y[t]]*k

                        # Vector of forecasts.
                        forecasts = [
                            float(forecasts_t1.loc[t-1, model_name]),
                            float(forecasts_t2.loc[t-2, model_name]),
                            float(forecasts_t3.loc[t-3, model_name]),
                        ]

                        # Collecting multi-value forecasting errors.
                        vector_errors = np.subtract(
                            forecasts, validation_vector)

                        # Saving necessary norms for later use.

                        # Regular p-norm.
                        p_norm_t = np.linalg.norm(
                            x=vector_errors, ord=p_norm)**p_norm
                        p_norm_df.loc[t, model_name] = p_norm_t

                        # Comparison p-norm.
                        MC_p_norm_df.loc[t, model_name] = p_norm_t/p_norm_g

        # Step 9: Define T_test.
        T_test = np.arange(T_train[2] + v + 1, 111)
        # Define T^{pi}, the index for profit calculation.
        T_pi = np.arange(T_train[2] + v + k + 2, 111)
        daily_profit_dict = create_H_tilde_dict(P, W)
        daily_profit_dict.update({"Long Only": {}})

        # Step 10: Adaptive Learning and Fixed model evaluation.
        for t in T_test:
            for p in P:
                for w in W:

                    # Step 10 (a): Collect MSE and daily profit for the fixed models.
                    model_name = naming_function(p, w)

                    # Obtain the forecast for the fixed model groups
                    forecast = float(forecasts_t3.loc[t, model_name])

                    # Obtain and save the MSE for the fixed model groups
                    MSE = (simulated_y[t+k] - forecast)**2

                    MSE_t[model_name][t] = MSE

                    # Obtain and save whether the direction is correct.
                    if np.sign(simulated_y[t+k]) == np.sign(forecast):
                        dir_correct_t[model_name] += 1/(len(T_test))

            # Step 10 (b): Collect MSE and daily profit for the fixed models.
            T_tilde = np.arange(t - v + 1, t + 1)

            # Step 10 (c): EN Learning.
            y_star, h_star = EN_learning(t,
                                         forecast_error_df,
                                         T_tilde,
                                         forecasts_t3,
                                         Lambda_vector)

            # Step 10 (d): EN Multi Learning.
            y_star_multi, h_star_multi = EN_learning(t,
                                                     p_norm_df,
                                                     T_tilde,
                                                     forecasts_t3,
                                                     Lambda_vector)

            # Step 10 (f): MC EN Learning.
            y_star_MC, h_star_MC = EN_learning(t,
                                               MC_p_norm_df,
                                               T_tilde,
                                               forecasts_t3,
                                               Lambda_vector)

            # Step 10 (g): Ensemble Learning, Ensemble Multi Learning and Ensemble MC Learning.
            y_star_ensemble, h_star_ensemble, y_star_ensemble_multi, h_star_ensemble_multi, y_star_ensemble_MC, h_star_ensemble_MC = All_ensembles(t,
                                                                                                                                                   v0,
                                                                                                                                                   v1,
                                                                                                                                                   Lambda_vector_Ensemble,
                                                                                                                                                   # We use t2 here because it gives us the model names without AL.
                                                                                                                                                   forecasts_t2.columns,
                                                                                                                                                   forecasts_t3,
                                                                                                                                                   forecast_error_df,
                                                                                                                                                   p_norm_df,
                                                                                                                                                   MC_p_norm_df)

            # EN: single-valued.
            MSE_EN = (simulated_y[t+k] - y_star)**2
            MSE_t['EN'][t] = MSE_EN
            forecasts_t3.loc[t, 'EN'] = y_star

            if np.sign(simulated_y[t+k]) == np.sign(y_star):
                dir_correct_t['EN'] += 1/(len(T_test))

            # EN: multi-valued.
            MSE_EN_multi = (simulated_y[t+k] - y_star_multi)**2
            MSE_t['EN Multi-Valued'][t] = MSE_EN_multi
            forecasts_t3.loc[t, 'EN Multi-Valued'] = y_star_multi
            if np.sign(simulated_y[t+k]) == np.sign(y_star_multi):
                dir_correct_t['EN Multi-Valued'] += 1/(len(T_test))

            # EN: MC
            MSE_EN_MC = (simulated_y[t+k] - y_star_MC)**2
            MSE_t['EN MC'][t] = MSE_EN_MC
            forecasts_t3.loc[t, 'EN MC'] = y_star_MC
            if np.sign(simulated_y[t+k]) == np.sign(y_star_MC):
                dir_correct_t['EN MC'] += 1/(len(T_test))

            # Ensemble: single-valued
            MSE_Ensemble = (simulated_y[t+k] - y_star_ensemble)**2
            MSE_t['Ensemble'][t] = MSE_Ensemble
            forecasts_t3.loc[t, 'Ensemble'] = y_star_ensemble
            if np.sign(simulated_y[t+k]) == np.sign(y_star_ensemble):
                dir_correct_t['Ensemble'] += 1/(len(T_test))

            # Ensemble: multi-valued
            MSE_Ensemble = (simulated_y[t+k] - y_star_ensemble_multi)**2
            MSE_t['Ensemble Multi-Valued'][t] = MSE_Ensemble
            forecasts_t3.loc[t,
                             'Ensemble Multi-Valued'] = y_star_ensemble_multi
            if np.sign(simulated_y[t+k]) == np.sign(y_star_ensemble_multi):
                dir_correct_t['Ensemble Multi-Valued'] += 1/(len(T_test))

            # Ensemble: MC
            MSE_Ensemble_MC = (
                simulated_y[t+k] - y_star_ensemble_MC)**2
            MSE_t['Ensemble MC'][t] = MSE_Ensemble_MC
            forecasts_t3.loc[t, 'Ensemble MC'] = y_star_ensemble_MC
            if np.sign(simulated_y[t+k]) == np.sign(y_star_ensemble_MC):
                dir_correct_t['Ensemble MC'] += 1/(len(T_test))

            model_selections_t["EN"][t] = h_star
            model_selections_t['EN Multi-Valued'][t] = h_star_multi
            model_selections_t['EN MC'][t] = h_star_MC
            model_selections_t['Ensemble'][t] = h_star_ensemble
            model_selections_t['Ensemble Multi-Valued'][t] = h_star_ensemble_multi
            model_selections_t['Ensemble MC'][t] = h_star_ensemble_MC

            # AIC and BIC
            for w in W:
                for criterion in ['bic', 'aic']:

                    model = auto_arima(y=simulated_y[(
                        t - w + 1): (t + 1)], max_p=3, start_d=0, max_d=1,  max_q=1, start_q=0, information_criterion=criterion)

                    p = int(str(model)[7])
                    d = int(str(model)[9])
                    q = 0

                    model_selections_t[f"{criterion.upper()} w{w}"][t] = "(" + \
                        f"{p}, {d}, {q}" + ")"

                    model = ARIMA(simulated_y[(t - w + 1): (t + 1)], order=(
                        p, d, q)).fit(method="yule_walker")

                    forecast = model.forecast(k)[-1]

                    MSE = (simulated_y[t+k] - forecast)**2

                    MSE_t[f"{criterion.upper()} w{w}"][t] = MSE

                    forecasts_t3.loc[t, f"{criterion.upper()} w{w}"] = forecast

                    if np.sign(simulated_y[t+k]) == np.sign(forecast):
                        dir_correct_t[f"{criterion.upper()} w{w}"] += 1 / \
                            (len(T_test))

            # Save whether long-only was directionally correct.
            if np.sign(simulated_y[t+k]) == 1:
                dir_correct_t["Long Only"] += 1/len(T_test)

            # Save the daily profit for all models of concern.

            if t in T_pi:
                for model in daily_profit_dict.keys():
                    if model != "Long Only":
                        daily_profit_dict[model][t] = trading_strategy(
                            forecasts_t3[model], simulated_S, k, t)
                    else:
                        daily_profit_dict["Long Only"][t] = trading_strategy(
                            None, simulated_S, k, t, True)

        daily_profit_dict_N[i] = daily_profit_dict

        # Step 11: Final Evaluation
        for model in daily_profit_dict.keys():
            if model != "Long Only":
                AMSE_N[model][i] = get_MSE(MSE_t, model, T_test)
            else:
                AMSE_N[model][i] = np.nan
            Sharpe_Ratios_N[model][i] = get_sharpe_ratio(
                daily_profit_dict, model)
            Annualised_Returns_N[model][i] = get_annualised_return(
                daily_profit_dict, model, T_pi)
            Max_DD_N[model][i] = get_max_dd(daily_profit_dict, model)
            perc_correct_N[model][i] = dir_correct_t[model]

        model_selections_N[i] = model_selections_t

    # Step 12: return the output
    with open(csv_directory + f'MSE_N_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(AMSE_N, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(csv_directory + f'Sharpes_N_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(Sharpe_Ratios_N, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(csv_directory + f'Ann_returns_N_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(Annualised_Returns_N, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open(csv_directory + f'MDDs_N_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(Max_DD_N, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(csv_directory + f'perc_correct_N_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(perc_correct_N, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(csv_directory + f'model_selections_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(model_selections_N, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open(csv_directory + f'time_series_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(time_series, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open(csv_directory + f'daily_prof_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(daily_profit_dict_N, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    with open(csv_directory + f't_simu_{N[0]}_{N[-1]}.pickle', 'wb') as handle:
        pickle.dump(t_simu_N, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    forecasts_t3.to_csv(csv_directory + 'forecast_df.csv')

    return [AMSE_N, Sharpe_Ratios_N, Annualised_Returns_N, Max_DD_N, model_selections_N, perc_correct_N, forecasts_t3]

### Adaptive Learning Functions ###


def EN_learning(t, forecast_error_df, T_tilde, forecast_df, Lambda_vector):

    errors_T_tilde = forecast_error_df.loc[T_tilde].apply(
        lambda x: exponential_learning(x, Lambda_vector))

    h_star = errors_T_tilde.idxmin(axis=1)

    y_star = forecast_df.loc[t, h_star]

    return y_star, h_star


def exponential_learning(forecast_errors, lambda_vector):
    return np.dot(forecast_errors, lambda_vector)


def All_ensembles(t, v0, v1, Lambda_vector, functional_sets, forecast_df, forecast_error_df, mutli_p_norm_df, MC_p_norm_df):

    T_0 = np.arange(t - v0 + 1, t+1)

    minimiser_count_single = create_value_dict(functional_sets)
    minimiser_count_multi = create_value_dict(functional_sets)
    minimiser_count_MC = create_value_dict(functional_sets)

    for r in T_0:

        T_1 = np.arange(r - v1 + 1, r+1)

        errors_T_1_single = forecast_error_df.loc[T_1].apply(
            lambda x: exponential_learning(x, Lambda_vector))
        h_star_r_single = errors_T_1_single.idxmin(axis=1)
        minimiser_count_single[h_star_r_single] += 1

        errors_T_1_multi = mutli_p_norm_df.loc[T_1].apply(
            lambda x: exponential_learning(x, Lambda_vector))
        h_star_r_multi = errors_T_1_multi.idxmin(axis=1)
        minimiser_count_multi[h_star_r_multi] += 1

        errors_T_1_MC = MC_p_norm_df.loc[T_1].apply(
            lambda x: exponential_learning(x, Lambda_vector))
        h_star_r_MC = errors_T_1_MC.idxmin(axis=1)
        minimiser_count_MC[h_star_r_MC] += 1

    forecasts_candidates = [float(forecast_df.loc[t, model])
                            for model in functional_sets]

    ensemble_weights_single = {model: count/len(T_0)
                               for model, count in minimiser_count_single.items()}

    ensembled_forecast_single = np.dot(
        list(ensemble_weights_single.values()), forecasts_candidates)

    ensemble_weights_multi = {model: count/len(T_0)
                              for model, count in minimiser_count_multi.items()}

    ensembled_forecast_multi = np.dot(
        list(ensemble_weights_multi.values()), forecasts_candidates)

    ensemble_weights_MC = {model: count/len(T_0)
                           for model, count in minimiser_count_MC.items()}

    ensembled_forecast_MC = np.dot(
        list(ensemble_weights_MC.values()), forecasts_candidates)

    return ensembled_forecast_single, ensemble_weights_single, ensembled_forecast_multi, ensemble_weights_multi, ensembled_forecast_MC, ensemble_weights_MC

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
    H_tilde_dict.update({'EN MC': {}})
    H_tilde_dict.update({'Ensemble MC': {}})
    for w in W:
        H_tilde_dict.update({f"AIC w{w}": {}})
        H_tilde_dict.update({f"BIC w{w}": {}})
    H_tilde_dict.update({'Long Only': {}})
    return H_tilde_dict


def create_all_value_dict(P, W):
    H_tilde_dict = {}
    for p in P:
        for w in W:
            H_tilde_dict.update({naming_function(p, w): 0})
    H_tilde_dict.update({"EN": 0})
    H_tilde_dict.update({"Ensemble": 0})
    H_tilde_dict.update({'EN Multi-Valued': 0})
    H_tilde_dict.update({'Ensemble Multi-Valued': 0})
    H_tilde_dict.update({'EN MC': 0})
    H_tilde_dict.update({'Ensemble MC': 0})
    for w in W:
        H_tilde_dict.update({f"AIC w{w}": 0})
        H_tilde_dict.update({f"BIC w{w}": 0})
    H_tilde_dict.update({'Long Only': 0})
    return H_tilde_dict


def create_value_dict(functional_sets):
    value_dict = {}
    for model in functional_sets:
        value_dict.update({model: 0})
    return value_dict


def create_model_df(T_train, functional_sets):
    headers = []
    for functional_set in functional_sets:
        if "en" not in functional_set.lower() and 'IC' not in functional_set and "Long" not in functional_set:
            headers.append(functional_set)
    forecast_df = df(columns=[headers])
    forecast_df['time_index'] = T_train
    forecast_df.set_index(T_train, inplace=True)
    del forecast_df['time_index']
    return forecast_df


def naming_function(p, w):
    return f"(AR{p}, w{w})"


def trading_strategy(model_forecasts, stock_price, k, t, long_only=False):

    signal_t_minus_1 = 0

    # Long if return prediction > 0 ; otherwise short.
    for i in range(1, k+1):

        if long_only == False:
            if float(model_forecasts.loc[t-i]) > 0:
                signal_t_minus_1 += 1
            elif float(model_forecasts.loc[t-i]) < 0:
                signal_t_minus_1 -= 1
        else:
            signal_t_minus_1 += 1

    PL_t = (1/k)*signal_t_minus_1 * \
        ((stock_price[t] - stock_price[t-1])/stock_price[t-1])

    return float(PL_t)


def get_sharpe_ratio(daily_profit_dict, model):
    mean = np.mean(list(daily_profit_dict[model].values()))
    std_dev = np.std(list(daily_profit_dict[model].values()))
    return 252**(0.5)*mean/std_dev


def get_MSE(error_dict, model, T_model):
    errors_as_array = list(error_dict[model].values())
    sum_squared_errors = sum(np.power(errors_as_array, 2))
    return (1/len(T_model))*sum_squared_errors


def get_annualised_return(daily_profit_dict, model, T_pi):
    return np.cumsum(list(daily_profit_dict[model].values()))[-1]*(252/len(T_pi))


def get_max_dd(daily_profit_dict, model):
    cumulative_profit = df(np.cumsum(list(daily_profit_dict[model].values())))
    rolling_max = (cumulative_profit+1).cummax()
    period_drawdown = (((1+cumulative_profit)/rolling_max) - 1).astype(float)
    drawdown = round(period_drawdown.min(), 3)
    return drawdown
