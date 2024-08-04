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
from tm_utils import Task


def extract_features(environment, config):
    ls = environment.storage._latent_similarities
    vs = np.zeros((config.OBJ_NUM, config.OBJ_NUM))
    g_indices = []
    p_indices = []
    h_indices = []
    for i in range(config.OBJ_NUM):
        for j in range(config.OBJ_NUM):
            vs[i, j] = environment.storage.get_visual_similarity(i, j)
        if environment.get_objects()[i].task == Task.GRASPING:
            g_indices.append(i)
        elif environment.get_objects()[i].task == Task.PUSHING:
            p_indices.append(i)
        else:
            h_indices.append(i)

    g_ls = ls[g_indices][:, g_indices]
    g_vs = vs[g_indices][:, g_indices]
    p_ls = ls[p_indices][:, p_indices]
    p_vs = vs[p_indices][:, p_indices]
    h_ls = ls[h_indices][:, h_indices]
    h_vs = vs[h_indices][:, h_indices]

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
    c = pearsonr(df["vs"], df["ls"])[0]
    if np.isnan(c):  # handle the nan case
        c = 0.0
    return float(c)


def get_spearmans_correlation_coefficient(df):
    c = spearmanr(df["vs"], df["ls"])[0]
    if np.isnan(c):  # handle the nan case
        c = 0.0
    return float(c)


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
    c = kendalltau(df["ls"], df["vs"])[0]
    if np.isnan(c):  # handle the nan case
        c = 0.0
    return float(c)


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

    # get_bland_altman_plots(means, diffs)

    return final_scores, final_scores_boolean


if __name__ == "__main__":
    config = Config()
    seed_num = 10
    scores_all = {}
    scores_b_all = {}
    for seed in range(seed_num):
        config.SEED = seed
        environment = Environment(config)
        scores, scores_b = run_across_tasks(config, environment)
        for k in scores.keys():
            if k not in scores_all:
                scores_all[k] = scores[k]
                scores_b_all[k] = scores_b[k]
            else:
                scores_all[k] += scores[k]
                scores_b_all[k] += scores_b[k]
    for k in scores_all.keys():
        scores_all[k] /= seed_num
        scores_b_all[k] /= seed_num

    pprint(scores_all)
    pprint(scores_b_all)

    # {'Concordance correlation coefficient': 0.3776022389862841,
    #  'DINOBot NN score': 0.26348039215686275,
    #  'Explained variance score': 0.1251009912037471,
    #  "Kendall's Tau": 0.30979073502632226,
    #  'Linear regression R^2': 0.16594272994625095,
    #  'MLP regression R^2': 0.13997365313931934,
    #  'Mean absolute error': 0.2669570360593219,
    #  "Pearson's correlation": 0.4738449902623226,
    #  'Random forest regression R^2': -0.08979552320542546,
    #  'Root mean squared error': 0.32561714268361175,
    #  "Spearman's correlation": 0.4253562628882091,
    #  'Support vector regression R^2': 0.09565472640924788,
    #  'Symmetric mean absolute percentage error': 79.25234737408707}
    # {'Concordance correlation coefficient': 0.2607639917926075,
    #  'DINOBot NN score': 0.3546666666666667,
    #  'Explained variance score': -0.24815689583829226,
    #  "Kendall's Tau": 0.3076446293744805,
    #  'Linear regression R^2': 0.07009095005627693,
    #  'MLP regression R^2': 0.04514351307184553,
    #  'Mean absolute error': 0.3859483131487889,
    #  "Pearson's correlation": 0.30764462937448056,
    #  'Random forest regression R^2': 0.070019534730163,
    #  'Root mean squared error': 0.6022111401425928,
    #  "Spearman's correlation": 0.30764462937448056,
    #  'Support vector regression R^2': -0.1380371580562025,
    #  'Symmetric mean absolute percentage error': 134.74684504356}
