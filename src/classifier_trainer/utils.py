import pandas as pd
from typing import List, Dict
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval


def _custom_eval(x) -> List:
    if str(x) == "nan":
        return []
    if str(x) == "[None]":
        return []
    if type(x) == list:
        return x
    else:
        return literal_eval(x)


def _preprocess_df(df: pd.DataFrame, classification_col: str) -> pd.DataFrame:
    classification_df = df.copy().drop_duplicates()
    """
    1. clean excerpt
    2. extract gender keywords
    3. extract country keywords
    """

    classification_df["excerpt"] = classification_df["excerpt"].apply(
        lambda x: x.replace("-", " ")
        .replace("’", "'")
        .replace("`", "'")
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace("[", " [ ")
        .replace("]", " ] ")
        .replace("—", " ")
        .replace("”", "'")
    )

    classification_df[classification_col] = classification_df[classification_col].apply(
        _custom_eval
    )

    return classification_df


def _hypertune_threshold(model, val_data, f_beta: float = 1):
    """
    having the probabilities, loop over a list of thresholds to see which one:
    1) yields the best results
    2) without being an aberrant value
    """

    logit_predictions, y_true = model.custom_predict(
        val_data,
        setup="hypertuning_threshold",
    )

    optimal_thresholds_dict = {}
    tagnames = list(model.tagname_to_tagid_classification.keys())

    for j in range(logit_predictions.shape[1]):
        preds_one_column = logit_predictions[:, j]
        min_proba = np.round(
            max(0.01, min(preds_one_column)), 3
        )  # so no value equal to 0
        max_proba = np.round(
            max(0.01, max(preds_one_column)), 3
        )  # so no value equal to 0

        thresholds_list = np.round(np.linspace(max_proba, min_proba, 21), 3)

        f_beta_scores = []
        precision_scores = []
        recall_scores = []
        for thresh_tmp in thresholds_list:
            score = _get_metrics_one_label(
                np.array(preds_one_column > thresh_tmp).astype(int),
                np.array(y_true[:, j]),
                f_beta,
            )
            f_beta_scores.append(score["f_score"])
            precision_scores.append(score["precision"])
            recall_scores.append(score["recall"])

        max_threshold = 0.01
        best_f_beta_score = -1

        for i in range(1, len(f_beta_scores) - 1):

            f_beta_score_mean = np.mean(f_beta_scores[i - 1 : i + 2])
            precision_score_mean = np.mean(precision_scores[i - 1 : i + 2])
            recall_score_mean = np.mean(recall_scores[i - 1 : i + 2])

            if (
                f_beta_score_mean >= best_f_beta_score
                and abs(recall_score_mean - precision_score_mean) < 0.4
            ):

                best_f_beta_score = f_beta_score_mean
                max_threshold = thresholds_list[i]

        tag_name = tagnames[j]

        optimal_thresholds_dict[tag_name] = max_threshold

    return optimal_thresholds_dict


def _generate_test_set_results(
    transformer_model,
    test_data: pd.DataFrame,
):
    """
    Generate test set results
    1- Generate predictions and results on labeled test set
    2- Get results on test set of project that contain the tag (to avoid having false negatives)
    For sector tags: No entries where there is 'Cross'.
    """
    # Generate predictions and results on labeled test set
    test_df = test_data.copy()
    predictions = transformer_model.custom_predict(
        test_df.excerpt.tolist(), setup="testing"
    )

    final_results_as_dict = _generate_classification_results(
        predictions,
        test_df.target_classification.tolist(),
        transformer_model.tagname_to_tagid_classification,
    )
    final_results_as_df = _get_results_df_from_dict(final_results_as_dict)
    return final_results_as_df


def _generate_classification_results(
    predictions: List[List[str]],
    groundtruth: List[List[str]],
    tagname_to_tagid: Dict[str, int],
):

    n_entries = len(predictions)
    n_tags = len(tagname_to_tagid)
    binary_outputs_predictions = np.zeros((n_entries, n_tags))
    binary_outputs_groundtruth = np.zeros((n_entries, n_tags))

    for i in range(n_entries):
        preds_one_entry = set(predictions[i])
        groundtruth_one_entry = set(groundtruth[i])

        for tagname, tagid in tagname_to_tagid.items():
            if tagname in preds_one_entry:
                binary_outputs_predictions[i, tagid] = 1

            if tagname in groundtruth_one_entry:
                binary_outputs_groundtruth[i, tagid] = 1

    tot_scores = {}

    for tagname, tagid in tagname_to_tagid.items():
        predictions_one_tag = binary_outputs_predictions[:, tagid]
        groundtruths_one_tag = binary_outputs_groundtruth[:, tagid]

        scores_one_tag = _get_metrics_one_label(
            predictions_one_tag, groundtruths_one_tag
        )

        tot_scores[tagname] = scores_one_tag

    return tot_scores


def _get_results_df_from_dict(final_results: Dict[str, Dict[str, float]]):
    """
    input: Dict: {tagname: {metric: score}}
    output: results as a dataframe and mean outputs of each tag
    """
    results_as_df = pd.DataFrame.from_dict(final_results, orient="index")
    metrics_list = list(results_as_df.columns)
    results_as_df["tag"] = results_as_df.index
    results_as_df.sort_values(by=["tag"], inplace=True, ascending=True)

    final_results_df = results_as_df.copy()

    # get mean results level 1
    mean_results_df = results_as_df.copy()
    mean_results_df["tag"] = mean_results_df["tag"].apply(
        lambda x: "mean->" + "->".join(x.split("->")[:-1])
    )
    mean_results_df = mean_results_df.groupby("tag", as_index=False).agg(
        {metric: lambda x: np.mean(list(x)) for metric in metrics_list}
    )
    mean_results_df["positive_examples_proportion"] = "-"
    final_results_df = pd.concat([final_results_df, mean_results_df])

    # get mean results level 0
    mean_results_df["tag"] = mean_results_df["tag"].apply(
        lambda x: "mean->" + x.split("->")[1]
    )
    mean_results_df = mean_results_df.groupby("tag", as_index=False).agg(
        {metric: lambda x: np.mean(list(x)) for metric in metrics_list}
    )
    mean_results_df["positive_examples_proportion"] = "-"
    final_results_df = pd.concat([final_results_df, mean_results_df])

    ordered_columns = ["tag"] + metrics_list + ["positive_examples_proportion"]
    final_results_df = final_results_df[ordered_columns].round(
        {col: 3 for col in ordered_columns if col != "tag"}
    )

    return final_results_df


def _get_metrics_one_label(preds: List[int], groundtruth: List[int], f_beta=1):
    """
    metrics for one tag
    """

    precision, recall, f_score, _ = precision_recall_fscore_support(
        groundtruth, preds, average="binary", beta=f_beta, zero_division=0
    )

    # confusion_results = confusion_matrix(groundtruth, preds, labels=[0, 1])
    # n_test_set_excerpts = sum(sum(confusion_results))
    # accuracy = (confusion_results[0, 0] + confusion_results[1, 1]) / n_test_set_excerpts
    # sensitivity = confusion_results[0, 0] / (
    #    confusion_results[0, 0] + confusion_results[0, 1]
    # )
    # specificity = confusion_results[1, 1] / (
    #    confusion_results[1, 0] + confusion_results[1, 1]
    # )

    results_as_dict = {
        "precision": np.round(precision, 3),
        "recall": np.round(recall, 3),
        "f_score": np.round(f_score, 3),
        # "accuracy": np.round(accuracy, 3),
        # "sensitivity": np.round(sensitivity, 3),
        # "specificity": np.round(specificity, 3),
    }

    return results_as_dict
