import sys

sys.path.append(".")
import multiprocessing
import logging
from pathlib import Path
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
import argparse
from ast import literal_eval
from pipeline import TrainingPipeline, BIASES_TYPES

EMBEDDING_DISTANCES_TYPES = ["cosine_distances", "euclidean_distances"]

logging.basicConfig(level=logging.INFO)

shift_cols = [
    "original_values_diff_mean_shift",
    "original_values_diff_absolute_shift",
    "absolute_values_diff_mean_shift",
    "absolute_values_diff_absolute_shift",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--hyperparameters", type=str)
    parser.add_argument("--architecture_setups", type=str)
    parser.add_argument("--training_setups", type=str)
    parser.add_argument("--backbone_names", type=str)
    # Data, model, and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument(
        "--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    # parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args, _ = parser.parse_known_args()

    whole_df = all_data = pd.read_pickle(f"{args.training_dir}/train.pickle")  # default
    train_df = whole_df[whole_df.excerpt_type == "train"].drop(columns=["excerpt_type"])
    val_df = whole_df[whole_df.excerpt_type == "val"].drop(columns=["excerpt_type"])
    test_df = whole_df[whole_df.excerpt_type == "test"].drop(columns=["excerpt_type"])

    architecture_setups = literal_eval(args.architecture_setups)
    training_setups = literal_eval(args.training_setups)
    backbone_names = literal_eval(args.backbone_names)
    hyperparameters = literal_eval(args.hyperparameters)

    relevant_classification_results_cols = [
        "mean->first_level_tags->pillars_1d",
        "mean->first_level_tags->pillars_2d",
        "mean->first_level_tags->sectors",
        "mean->subpillars_1d",
        "mean->subpillars_2d",
    ]
    final_cols_classification = [
        "backbone_name",
        "training_setup",
        "architecture_setup",
        "tag",
        "precision",
        "f_score",
    ]
    classification_results_df = pd.DataFrame()
    overall_results_df = pd.DataFrame()

    for one_backbone_name in backbone_names:
        for one_training_setup in training_setups:
            for one_architecture_setup in architecture_setups:
                pipeline = TrainingPipeline(
                    hyperparameters,
                    train_df,
                    val_df,
                    test_df,
                    architecture_setup=one_architecture_setup,
                    training_setup=one_training_setup,
                    backbone_name=one_backbone_name,
                    results_dir=Path(args.output_data_dir) / "results",
                    humbias_set_dir=Path(args.output_data_dir) / "humbias_set",
                )
                if one_training_setup == "no_finetuning":
                    pipeline.test_set_results = pd.DataFrame(
                        {
                            "tag": list(
                                pipeline.model.tagname_to_tagid_classification.keys()
                            )
                        }
                    )
                    pipeline.test_set_results["precision"] = "-"
                    pipeline.test_set_results["f_score"] = "-"

                relevant_classification_results = pipeline.test_set_results
                relevant_classification_results = relevant_classification_results[
                    relevant_classification_results.tag.isin(
                        relevant_classification_results_cols
                    )
                ].copy()
                relevant_classification_results["backbone_name"] = one_backbone_name
                relevant_classification_results["training_setup"] = one_training_setup
                relevant_classification_results[
                    "architecture_setup"
                ] = one_architecture_setup

                relevant_classification_results = relevant_classification_results[
                    final_cols_classification
                ]
                classification_results_df = pd.concat(
                    [classification_results_df, relevant_classification_results]
                )

                if one_training_setup != "no_finetuning":
                    # counterfactual_predictions_discrepency
                    pipeline.get_counterfactual_predictions_discrepency_results()

                    # counterfactual explainability discrepency
                    pipeline.get_counterfactual_explainability_discrepency_results()

                # counterfactual embeddings discrepency
                pipeline.get_counterfactual_embeddings_discrepency_results()

                # probing results
                pipeline.get_probing_results()

                added_row = {
                    "backbone_name": one_backbone_name,
                    "training_setup": one_training_setup,
                    "architecture_setup": one_architecture_setup,
                    "mean_precision": relevant_classification_results.precision.mean(),
                    "mean_f_score": relevant_classification_results.f_score.mean(),
                }
                for one_bias_type in BIASES_TYPES:
                    for one_shift_col in shift_cols:
                        if one_training_setup != "no_finetuning":
                            added_row[
                                f"{one_bias_type}_{one_shift_col}_counterfactual"
                            ] = (
                                pipeline.output_counterfactual_predictions_discrepency_results[
                                    one_bias_type
                                ][
                                    one_shift_col
                                ]
                                .apply(abs)
                                .sum()
                                * 100
                                / len(pipeline.model.tagname_to_tagid_classification)
                            )

                            added_row[
                                f"{one_bias_type}_{one_shift_col}_explainability"
                            ] = (
                                pipeline.output_explainability_results[one_bias_type][
                                    one_shift_col
                                ]
                                .apply(abs)
                                .sum()
                                # * 100
                                / len(pipeline.model.tagname_to_tagid_classification)
                            )

                        for one_embedding_dist in EMBEDDING_DISTANCES_TYPES:
                            added_row[
                                f"{one_bias_type}_{one_shift_col}_{one_embedding_dist}_embeddings"
                            ] = (
                                pipeline.embedding_results[one_bias_type][
                                    one_embedding_dist
                                ][one_shift_col].sum()
                                # * 100
                                / len(pipeline.model.tagname_to_tagid_classification)
                            )

                overall_results_df = overall_results_df.append(
                    added_row, ignore_index=True
                )

    overall_results_df.to_csv(
        Path(args.output_data_dir) / "results" / "overall_results.csv", index=None
    )
    classification_results_df.to_csv(
        Path(args.output_data_dir) / "results" / "classification_results.csv",
        index=None,
    )
