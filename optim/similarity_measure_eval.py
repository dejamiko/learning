from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    f1_score,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from config import Config
from playground.environment import Environment


def extract_features(environment, config):
    ls = environment.storage._latent_similarities
    vs = np.zeros((config.OBJ_NUM, config.OBJ_NUM))
    for i in range(config.OBJ_NUM):
        for j in range(config.OBJ_NUM):
            vs[i, j] = environment.storage.get_visual_similarity(i, j)

    start_g, stop_g = 0, 16
    start_p, stop_p = 17, 34
    start_h, stop_h = 35, 51
    g_ls = ls[start_g:stop_g, start_g:stop_g]
    g_vs = vs[start_g:stop_g, start_g:stop_g]
    p_ls = ls[start_p:stop_p, start_p:stop_p]
    p_vs = vs[start_p:stop_p, start_p:stop_p]
    h_ls = ls[start_h:stop_h, start_h:stop_h]
    h_vs = vs[start_h:stop_h, start_h:stop_h]

    return (
        (g_ls, p_ls, h_ls),
        (g_vs, p_vs, h_vs),
        ("grasping", "pushing", "hammering"),
    )


def convert_to_bool(s, thresh):
    s = s.where(s < thresh, 1.0)
    s = s.where(s >= thresh, 0.0)
    return s


def get_boolean_df(df, config):
    b_df = df.copy(deep=True)
    b_df["ls"] = convert_to_bool(df["ls"], config.PROB_THRESHOLD)
    b_df["vs"] = convert_to_bool(df["vs"], config.SIMILARITY_THRESHOLD)
    return b_df


def get_pearsons_correlation_coefficient(df):
    return float(pearsonr(df["vs"], df["ls"])[0])


def get_spearmans_correlation_coefficient(df):
    return float(spearmanr(df["vs"], df["ls"])[0])


def get_f1_score(df):
    return float(f1_score(df["ls"], df["vs"]))


def get_cv_r2(df, model):
    return float(
        cross_val_score(model, df[["vs"]], df["ls"], cv=5, scoring="r2").mean()
    )


def get_concordance_correlation_coefficient(df):
    mean_true = np.mean(df["ls"])
    mean_pred = np.mean(df["vs"])
    var_true = np.var(df["ls"])
    var_pred = np.var(df["vs"])
    covariance = np.mean((df["ls"] - mean_true) * (df["vs"] - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return float(ccc)


def get_explained_var_score(df):
    return explained_variance_score(df["ls"], df["vs"])


def get_mae(df):
    return float(mean_absolute_error(df["ls"], df["vs"]))


def get_rmse(df):
    return float(np.sqrt(mean_squared_error(df["ls"], df["vs"])))


def get_smape(df):
    return float(
        np.mean(
            2.0 * np.abs(df["vs"] - df["ls"]) / (np.abs(df["vs"]) + np.abs(df["ls"]))
        )
        * 100
    )


def get_kendalls_tau(df):
    return float(kendalltau(df["ls"], df["vs"])[0])


def get_dinobot_nn_metric(df):
    vs = np.array(df["vs"]).reshape(int(np.sqrt(len(df["vs"]))), -1)
    ls = np.array(df["ls"]).reshape(int(np.sqrt(len(df["ls"]))), -1)
    count_correct = 0
    count_all = 0
    for i, (row_l, row_v) in enumerate(zip(ls, vs)):
        nns_l = find_nn_not_self(row_l, i)
        nns_v = find_nn_not_self(row_v, i)

        for nn_c in nns_v:
            if nn_c in nns_l:
                count_correct += 1
            count_all += 1
    return count_correct / count_all


def find_nn_not_self(array, index):
    nns = []

    best_sim = -1
    for i in range(len(array)):
        if index == i:
            continue
        if array[i] > best_sim:
            best_sim = array[i]
            nns = [i]
        elif array[i] == best_sim:
            nns.append(i)

    return nns


def get_residual_analysis_plot(df, model):
    model.fit(df[["vs"]], df["ls"])

    y_pred = model.predict(df[["vs"]])
    residuals = df["ls"] - y_pred

    plt.scatter(y_pred, residuals)
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residual Analysis")
    return plt


def get_bland_altman_plots(means, diffs):
    fig, axes = plt.subplots(1, len(means))

    for m, d, axis in zip(means, diffs, axes):
        mean_diff = np.mean(d)
        std_diff = np.std(d)

        fig.suptitle("Bland-Altman Plots")
        axis.scatter(m, d)
        axis.axhline(mean_diff, color="red", linestyle="--")
        axis.axhline(mean_diff + 1.96 * std_diff, color="grey", linestyle="--")
        axis.axhline(mean_diff - 1.96 * std_diff, color="grey", linestyle="--")
    plt.show()


def run_full_suite(df):
    scores = {
        # metrics that fit some model or allow for the features to have some relation
        # linear relation
        "Linear regression R^2": get_cv_r2(df, LinearRegression()),
        "Pearson's correlation": get_pearsons_correlation_coefficient(df),
        # non-linear relation
        "Random forest regression R^2": get_cv_r2(df, RandomForestRegressor()),
        "Support vector regression R^2": get_cv_r2(df, SVR()),
        "MLP regression R^2": get_cv_r2(
            df, MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
        ),
        "Spearman's correlation": get_spearmans_correlation_coefficient(df),
        "Kendall's Tau": get_kendalls_tau(df),
        "Explained variance score": get_explained_var_score(df),
        "Concordance correlation coefficient": get_concordance_correlation_coefficient(
            df
        ),
        # metrics that compare the scores directly
        "Mean absolute error": get_mae(df),
        "Root mean squared error": get_rmse(df),
        "Symmetric mean absolute percentage error": get_smape(df),
        # application specific (arguably most important)
        "DINOBot NN score": get_dinobot_nn_metric(df),
    }
    return scores


def run_across_tasks(config, environment):
    ls_all, vs_all, tasks = extract_features(environment, config)
    scores = []
    scores_boolean = []
    means, diffs = [], []
    for vs, ls, task in zip(vs_all, ls_all, tasks):
        assert np.allclose(
            vs, vs.flatten().reshape(int(np.sqrt(len(vs.flatten()))), -1)
        )
        df = pd.DataFrame({"vs": vs.flatten(), "ls": ls.flatten()})
        real_score = run_full_suite(df)
        means.append(np.mean([df["vs"], df["ls"]], axis=0))
        diffs.append(df["vs"] - df["ls"])
        boolean_score = run_full_suite(get_boolean_df(df, config))
        scores.append(real_score)
        scores_boolean.append(boolean_score)
    final_scores = {}
    for s in scores:
        for k in s.keys():
            if k not in final_scores:
                final_scores[k] = 0
            final_scores[k] += s[k]
    for k in final_scores.keys():
        final_scores[k] /= len(scores)

    final_scores_boolean = {}
    for s in scores_boolean:
        for k in s.keys():
            if k not in final_scores_boolean:
                final_scores_boolean[k] = 0
            final_scores_boolean[k] += s[k]
    for k in final_scores_boolean.keys():
        final_scores_boolean[k] /= len(scores_boolean)

    get_bland_altman_plots(means, diffs)

    return final_scores, final_scores_boolean


if __name__ == "__main__":
    config = Config()
    environment = Environment(config)
    scores, scores_b = run_across_tasks(config, environment)
    pprint(scores)
    pprint(scores_b)

    # {'Concordance correlation coefficient': 0.3647922077644788,
    #  'DINOBot NN score': 0.24387254901960786,
    #  'Explained variance score': 0.07950940150307313,
    #  "Kendall's Tau": 0.29935260372914635,
    #  'Linear regression R^2': 0.14344368069746866,
    #  'MLP regression R^2': 0.1589571371315341,
    #  'Mean absolute error': 0.27657619516019727,
    #  "Pearson's correlation": 0.45332414984185765,
    #  'Random forest regression R^2': -0.07635235542928556,
    #  'Root mean squared error': 0.3362796901726665,
    #  "Spearman's correlation": 0.4106749924916196,
    #  'Support vector regression R^2': 0.1389715609866361,
    #  'Symmetric mean absolute percentage error': 79.7025055480695}
    # For the boolean version
    # {'Concordance correlation coefficient': 0.27065814110136194,
    #  'DINOBot NN score': 0.3642297663350295,
    #  'Explained variance score': -0.25140203599268157,
    #  "Kendall's Tau": 0.31561097197470167,
    #  'Linear regression R^2': 0.07999039210586031,
    #  'MLP regression R^2': 0.06609067621115317,
    #  'Mean absolute error': 0.3842362312572088,
    #  "Pearson's correlation": 0.3156109719747017,
    #  'Random forest regression R^2': 0.08091835008868524,
    #  'Root mean squared error': 0.6012933324275102,
    #  "Spearman's correlation": 0.3156109719747017,
    #  'Support vector regression R^2': -0.22381205486484756,
    #  'Symmetric mean absolute percentage error': 132.53164303452695}
