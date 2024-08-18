import json
import os.path
import time

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
from tm_utils import (
    Task,
    ImageEmbeddings,
    SimilarityMeasure,
    ContourImageEmbeddings,
    ContourSimilarityMeasure,
    ImagePreprocessing,
    NNSimilarityMeasure,
    NNImageEmbeddings,
)

metrics_is_larger_better = {
    "Linear regression R^2": True,
    "Pearson's correlation": True,
    "Random forest regression R^2": True,
    "Support vector regression R^2": True,
    "MLP regression R^2": True,
    "Spearman's correlation": True,
    "Kendall's Tau": True,
    "Explained variance score": True,
    "Concordance correlation coefficient": True,
    "Mean absolute error": False,
    "Root mean squared error": False,
    "Symmetric mean absolute percentage error": False,
    "DINOBot NN score": True,
}

metrics_scale_0_1 = {
    "Linear regression R^2": lambda x: (x + 1) / 2,
    "Pearson's correlation": lambda x: (x + 1) / 2,
    "Random forest regression R^2": lambda x: (x + 1) / 2,
    "Support vector regression R^2": lambda x: (x + 1) / 2,
    "MLP regression R^2": lambda x: (x + 1) / 2,
    "Spearman's correlation": lambda x: (x + 1) / 2,
    "Kendall's Tau": lambda x: (x + 1) / 2,
    "Explained variance score": lambda x: max(0, (1 + x) / 2),
    "Concordance correlation coefficient": lambda x: (x + 1) / 2,
    "Mean absolute error": lambda x: 1 - x / 2,
    "Root mean squared error": lambda x: 1 - x / 2,
    "Symmetric mean absolute percentage error": lambda x: 1 - x / 200,
    "DINOBot NN score": lambda x: x,
}

metrics_weights = {
    "Pearson's correlation": 0.16,
    "Linear regression R^2": 0.16,
    "Explained variance score": 0.12,
    "Concordance correlation coefficient": 0.12,
    "DINOBot NN score": 0.12,
    "Random forest regression R^2": 0.04,
    "Support vector regression R^2": 0.04,
    "MLP regression R^2": 0.04,
    "Spearman's correlation": 0.04,
    "Kendall's Tau": 0.04,
    "Mean absolute error": 0.04,
    "Root mean squared error": 0.04,
    "Symmetric mean absolute percentage error": 0.04,
}


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
        (Task.GRASPING, Task.PUSHING, Task.HAMMERING),
    )


def convert_to_bool(s, thresh):
    s = s.where(s < thresh, 1.0)
    s = s.where(s >= thresh, 0.0)
    return s


def get_boolean_df(df, config, task):
    b_df = df.copy(deep=True)
    b_df["ls"] = convert_to_bool(df["ls"], config.PROB_THRESHOLD)
    b_df["vs"] = convert_to_bool(
        df["vs"], config.SIMILARITY_THRESHOLDS[Task.get_ind(task)]
    )
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
        boolean_score = run_full_suite(get_boolean_df(df, config, task))
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


def run_eval_one_config(config, n=10):
    scores_all = {}
    scores_b_all = {}
    for seed in range(n):
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
        scores_all[k] /= n
        scores_b_all[k] /= n
    return scores_all, scores_b_all


def float_comp(tally):
    if tally < 0:
        return -1
    elif tally > 0:
        return 1
    else:
        return 0


def compare_counts(scores_1, scores_2):
    sc_1, _ = scores_1
    sc_2, _ = scores_2

    tally = 0
    for metric in metrics_is_larger_better.keys():
        if sc_1[metric] > sc_2[metric]:
            if metrics_is_larger_better[metric]:
                tally += 1
            else:
                tally -= 1
        elif sc_1[metric] < sc_2[metric]:
            if not metrics_is_larger_better[metric]:
                tally += 1
            else:
                tally -= 1
    return tally


def compare_sums(scores_1, scores_2):
    sc_1, _ = scores_1
    sc_2, _ = scores_2

    tally = 0.0
    for m in metrics_is_larger_better.keys():
        if not metrics_is_larger_better[m]:
            tally += 1 / (1 + sc_1[m])
            tally -= 1 / (1 + sc_2[m])
        else:
            tally += sc_1[m]
            tally -= sc_2[m]

    return float_comp(tally)


def compare_sums_scaled(scores_1, scores_2):
    sc_1, _ = scores_1
    sc_2, _ = scores_2

    tally = 0.0
    for m in metrics_scale_0_1.keys():
        tally += metrics_scale_0_1[m](sc_1[m])
        tally -= metrics_scale_0_1[m](sc_2[m])

    return float_comp(tally)


def compare_dinobot_nn(scores_1, scores_2):
    sc_1, _ = scores_1
    sc_2, _ = scores_2

    return float_comp(sc_1["DINOBot NN score"] - sc_2["DINOBot NN score"])


def compare_weighted_sum(scores_1, scores_2):
    sc_1, _ = scores_1
    sc_2, _ = scores_2

    tally = 0.0
    for m in metrics_scale_0_1.keys():
        tally += metrics_weights[m] * metrics_scale_0_1[m](sc_1[m])
        tally -= metrics_weights[m] * metrics_scale_0_1[m](sc_2[m])

    return float_comp(tally)


def run_and_save(config, filename, n=10):
    all_scores = {}
    for emb in ImageEmbeddings:
        config.IMAGE_EMBEDDINGS = emb
        for sim in SimilarityMeasure:
            config.SIMILARITY_MEASURE = sim
            scores, scores_b = run_eval_one_config(config, n)
            all_scores[str(config)] = [scores, scores_b]
    for emb in ContourImageEmbeddings:
        config.IMAGE_EMBEDDINGS = emb
        for sim in ContourSimilarityMeasure:
            config.SIMILARITY_MEASURE = sim
            scores, scores_b = run_eval_one_config(config, n)
            all_scores[str(config)] = [scores, scores_b]
    config.IMAGE_EMBEDDINGS = NNImageEmbeddings.SIAMESE
    for sim in NNSimilarityMeasure:
        config.SIMILARITY_MEASURE = sim
        scores, scores_b = run_eval_one_config(config, n)
        all_scores[str(config)] = [scores, scores_b]
    if os.path.exists(filename):
        with open(filename, "r") as f:
            previous = json.load(f)
            for k, v in previous.items():
                if k not in all_scores:
                    all_scores[k] = v
    with open(filename, "w") as f:
        json.dump(all_scores, f)


def load_results(filename):
    with open(filename) as f:
        data = json.load(f)

    all_results = []
    all_results_b = []
    for k in data.keys():
        all_results.append((data[k][0], k))
        all_results_b.append((data[k][1], k))

    return all_results, all_results_b


if __name__ == "__main__":
    processing_steps_to_try = [
        [],
        [ImagePreprocessing.GREYSCALE],
        [ImagePreprocessing.BACKGROUND_REM],
        [ImagePreprocessing.CROPPING],
        [ImagePreprocessing.SEGMENTATION],
        [ImagePreprocessing.CROPPING, ImagePreprocessing.BACKGROUND_REM],
        [ImagePreprocessing.CROPPING, ImagePreprocessing.GREYSCALE],
        [
            ImagePreprocessing.CROPPING,
            ImagePreprocessing.BACKGROUND_REM,
            ImagePreprocessing.GREYSCALE,
        ],
    ]
    obj_num = 40
    run_num = 10
    for ps in processing_steps_to_try:
        start = time.time()
        config = Config()
        config.OBJ_NUM = obj_num
        config.IMAGE_PREPROCESSING = ps
        run_and_save(
            config,
            f"analysis/results/results_one_image_{config.OBJ_NUM}_{ps}.json",
            run_num,
        )
        config = Config()
        config.OBJ_NUM = obj_num
        config.IMAGE_PREPROCESSING = ps
        config.USE_ALL_IMAGES = True
        run_and_save(
            config,
            f"analysis/results/results_all_images_{config.OBJ_NUM}_{ps}.json",
            run_num,
        )
        print(f"Finished with {ps} in {time.time() - start}")

    # results, results_b = load_results("analysis/results_one_image_51.json")
    # results, results_b = load_results("analysis/results_all_images_51.json")
    #
    # results_sorted = sorted(results, key=cmp_to_key(compare_counts), reverse=True)
    # pprint(results_sorted[:4])
    #
    # print("=" * 50)
    # print("=" * 50)
    # print("Compare sums")
    # print("=" * 50)
    # print("=" * 50)
    #
    # results_sorted = sorted(results, key=cmp_to_key(compare_sums), reverse=True)
    # pprint(results_sorted[:4])
    #
    # print("=" * 50)
    # print("=" * 50)
    # print("Compare sums scaled")
    # print("=" * 50)
    # print("=" * 50)
    #
    # results_sorted = sorted(results, key=cmp_to_key(compare_sums_scaled), reverse=True)
    # pprint(results_sorted[:4])
    #
    # print("=" * 50)
    # print("=" * 50)
    # print("Compare weighted sum")
    # print("=" * 50)
    # print("=" * 50)
    #
    # results_sorted = sorted(results, key=cmp_to_key(compare_weighted_sum), reverse=True)
    # pprint(results_sorted[:4])
    #
    # print("=" * 50)
    # print("=" * 50)
    # print("Compare DINOBot NN")
    # print("=" * 50)
    # print("=" * 50)
    #
    # results_sorted = sorted(results, key=cmp_to_key(compare_dinobot_nn), reverse=True)
    # pprint(results_sorted[:4])
