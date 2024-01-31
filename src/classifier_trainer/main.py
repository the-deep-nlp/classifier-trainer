import os
import multiprocessing
import pandas as pd
import torch
import pytorch_lightning as pl
from typing import List, Dict
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoTokenizer

from .models.model import TrainingTransformer, Model
from .utils import _hypertune_threshold, _generate_test_set_results, _preprocess_df

os.environ["TOKENIZERS_PARALLELISM"] = "false"

architecture_setups = ["base_architecture", "multiabel_architecture"]
relevant_classification_columns = ["excerpt", "target_classification"]


class NoSavingCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def _del_model(self, *_):
        pass

    def _save_model(self, *_):
        pass


class ClassifierTrainer:
    def __init__(self):
        pass

    def _initialize_training_args(self):

        if torch.cuda.is_available():
            n_gpu = 1
            accelerator = "gpu"
            training_device = "cuda"
        else:
            n_gpu = 0
            accelerator = "cpu"
            training_device = "cpu"

        checkpoint_callback_params = {
            "save_top_k": 1,
            "verbose": True,
            "monitor": "val_loss",
            "mode": "min",
        }

        train_params = {
            "batch_size": self.hyperparameters["train_batch_size"],
            "shuffle": True,
            "num_workers": 4,
        }

        val_params = {
            "batch_size": self.hyperparameters["val_batch_size"],
            "shuffle": False,
            "num_workers": 4,
        }

        # self.classification_column = self.classification_column
        self.MODEL_DIR = "models"
        self.MODEL_NAME = f"trained_classifier_{self.results_name}.pt"

        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)

        if self.hyperparameters["enable_checkpointing"]:
            self.checkpoint_callback = ModelCheckpoint(
                dirpath=self.MODEL_DIR,
                filename=self.MODEL_NAME,
                **checkpoint_callback_params,
            )
        else:
            self.checkpoint_callback = NoSavingCheckpoint(
                monitor="val_loss", mode="min"
            )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=2, mode="min"
        )
        self.trainer = pl.Trainer(
            logger=None,
            callbacks=[early_stopping_callback, self.checkpoint_callback],
            enable_progress_bar=True,
            profiler="simple",
            # log_gpu_memory=True,
            enable_model_summary=True,
            accelerator=accelerator,
            precision=16 if n_gpu > 0 else 32,
            accumulate_grad_batches=1,
            max_epochs=self.hyperparameters["n_epochs"],
            gradient_clip_val=1,
            gradient_clip_algorithm="norm",
        )

        self.training_model = TrainingTransformer(
            model_name_or_path=self.backbone_name,
            train_dataset=self.train_df.copy(),
            val_dataset=self.val_df.copy(),
            train_params=train_params,
            val_params=val_params,
            tokenizer=self.tokenizer,
            plugin="deepspeed_stage_3_offload",
            accumulate_grad_batches=1,
            max_epochs=self.hyperparameters["n_epochs"],
            dropout_rate=self.hyperparameters["dropout"],
            weight_decay=self.hyperparameters["weight_decay"],
            learning_rate=self.hyperparameters["learning_rate"],
            max_len=self.hyperparameters["max_len"],
            n_freezed_layers=self.hyperparameters["n_freezed_layers"],
            loss_gamma=self.hyperparameters["loss_gamma"],
            proportions_pow=self.hyperparameters["proportions_pow"],
            architecture_setup=self.architecture_setup,
            training_device=training_device,
        )

    def train_classification_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        architecture_setup: str = "multiabel_architecture",
        backbone_name: str = "nlp-thedeep/humbert",
        results_dir: str = "results",
        enable_checkpointing: bool = True,
        save_model: bool = True,
        n_epochs: int = 3,
        dropout: float = 0.2,
        weight_decay: float = 1e-2,
        learning_rate: float = 5e-5,
        max_len: int = 200,
        n_freezed_layers: int = 1,
        train_batch_size: int = 8,
        val_batch_size: int = 16,
        loss_gamma: float = 1,
        proportions_pow: float = 0.5,
    ):
        """
        Inputs:
            - train_df: pd.DataFrame: Training DataFrame with 2 columns: ["excerpt", "target_classification"]
            - val_df: pd.DataFrame: Validation DataFrame with 2 columns: ["excerpt", "target_classification"]
            - architecture_setup: str: one of ["base_architecture", "multiabel_architecture"]
            - backbone_name: str: Backbone Name in HuggingFace
            - results_dir: str: Results directory
            - enable_checkpointing: bool: Whether or not to save model checkpoints while training
            - save_model: bool: Whether or not to save the model after training
            - n_epochs: int: Number of total training epochs
            - dropout: float: Dropout rate
            - weight_decay: float: AdamWeight weight decay
            - learning_rate: float: Training Learning Rate
            - max_len: int: Max entry input length
            - n_freezed_layers: int: Number of freezed layers in the model training
            - train_batch_size: int: Training batch size
            - val_batch_size: int: Validation batch size
            - loss_gamma: float: Gamma parameter in the focal loss
            - proportions_pow: float: Alpha parameter in the focal loss
        """
        for one_col in relevant_classification_columns:
            for df_type, one_df in {"train_df": train_df, "val_df": val_df}.items():
                assert (
                    one_col in one_df.columns
                ), f"'{df_type}' foes not contain the column '{one_col}'."

        self.hyperparameters = {
            "n_epochs": n_epochs,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "max_len": max_len,
            "n_freezed_layers": n_freezed_layers,
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
            "enable_checkpointing": enable_checkpointing,
            "proportions_pow": proportions_pow,
            "loss_gamma": loss_gamma,
        }
        assert (
            architecture_setup in architecture_setups
        ), f"arg 'architecture_setup' must be in {architecture_setups}"

        self.architecture_setup = architecture_setup
        self.backbone_name = backbone_name

        self.RESULTS_DIR = results_dir
        if not os.path.exists(self.RESULTS_DIR):
            os.makedirs(self.RESULTS_DIR)

        self.results_name = f"{backbone_name}_{self.architecture_setup}"

        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)

        self.train_df = _preprocess_df(train_df)
        self.val_df = _preprocess_df(val_df)

        self._initialize_training_args()
        self.trainer.fit(self.training_model)

        self.model = Model(self.training_model)
        self.model.optimal_thresholds = _hypertune_threshold(self.model, self.val_df)

        if save_model:
            torch.save(self.model, os.path.join(self.MODEL_DIR, self.MODEL_NAME))

        return self.model

    def load_model(self, model_path: str):
        """
        Inputs:
            - model_path: str: path of the saved model
        """
        self.model = torch.load(model_path)
        self.model.eval()
        return self.model

    def generate_test_predictions(self, sentences: List[str]) -> List[List[str]]:
        """
        Generate model predictions
        Inputs:
            - sentences: List[str]: List of entries
        Outputs:
            Relevant Tags: [[tag1, tag2, ...], [tag2, tag3], [...], ...]
        """
        assert (
            type(sentences) is list
        ), f"'sentences' argument must be a list of entries."
        assert hasattr(
            self, "model"
        ), f"no attribute 'model'. Please train your model using the 'train_classification_model' function or load it using the 'load_model' function."
        predictions = self.model.custom_predict(sentences, setup="testing")
        return predictions

    def generate_test_results(
        self,
        test_df: pd.DataFrame,
        generate_visulizations: bool = True,
        save_results: bool = True,
        results_dir: str = "results",
    ):
        """

        Outputs:
            - test_df: pd.DataFrame: test set results
            - generate_visulizations: bool: Whether to generate visualisations of the test results
            - save_results: bool: Whether to save results
        """
        assert hasattr(
            self, "model"
        ), f"no attribute 'model'. Please train your model using the 'train_classification_model' function or load it using the 'load_model' function."
        for one_col in relevant_classification_columns:
            assert (
                one_col in test_df.columns
            ), f"'test_df' foes not contain the column '{one_col}'."
        test_df = _preprocess_df(test_df)
        self.test_set_results = _generate_test_set_results(
            self.model, test_df
        ).sort_values(by="tag")

        if save_results:
            self.RESULTS_DIR = self.results_dir
            CLASSIFICATION_RESULTS_DIR = os.path.join(
                self.RESULTS_DIR, "classification"
            )
            if not os.path.exists(CLASSIFICATION_RESULTS_DIR):
                os.makedirs(CLASSIFICATION_RESULTS_DIR)
            file_name = f"classification_results.csv"
            self.test_set_results.to_csv(
                os.path.join(CLASSIFICATION_RESULTS_DIR, file_name)
            )

        if generate_visulizations:
            ...
            # TODO: generate visulaizations automatically for all tasks.

        return self.test_set_results
