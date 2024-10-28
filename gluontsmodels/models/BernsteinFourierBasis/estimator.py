# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Optional, Iterable, Dict, Any, List, Type

import torch
import lightning.pytorch as pl
import numpy as np
import pandas as pd

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.transform import (
    Transformation,
    AddObservedValuesIndicator,
    InstanceSampler,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    SelectFields,
    VstackFeatures,
)
from gluonts.time_feature import week_of_year_index
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import Output, StudentTOutput
from gluonts.torch.util import resolve_device


from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature

def target_transformation_length(
    target: np.ndarray, pred_length: int, is_train: bool
) -> int:
    return target.shape[-1] + (0 if is_train else pred_length)

from lightning_module import BasisLightningModule

PREDICTION_INPUT_NAMES = ["past_target", "past_observed_values", "past_feat_dynamic_real"]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]

class BasisEstimator(PyTorchLightningEstimator):
    """
    An estimator training the basis model. This combines a learned monotonic bernstein basis
    w/ a fourier basis.

    This class is uses the model defined in ``BasisModel``,
    and wraps it into a ``BasisLightningModule`` for training
    purposes: training is performed using PyTorch Lightning's ``pl.Trainer``
    class.

    Parameters
    ----------
    prediction_length
        Length of the prediction horizon.
    context_length
        Number of time steps prior to prediction time that the model
        takes as inputs (default: ``10 * prediction_length``).
    input_dim
        Size to project the input data.
    hidden_dim
        Hidden dimension in MLPs
    num_layers
        Number of hidden layers in the MLPs
    fourier_v
        Number of sins/cos in the fourier basis
    max_period
        Max period of the fourier basis
    bernstein_v
        Number/Order of the bernstein basis.
    lr
        Learning rate (default: ``1e-3``).
    weight_decay
        Weight decay regularization parameter (default: ``1e-8``).

    distr_output
        Distribution to use to evaluate observations and sample predictions
        (default: StudentTOutput()).
    batch_size
        The size of the batches to be used for training (default: 32).
    num_batches_per_epoch
        Number of batches to be processed in each training epoch
            (default: 50).
    trainer_kwargs
        Additional arguments to provide to ``pl.Trainer`` for construction.
    train_sampler
        Controls the sampling of windows during training.
    validation_sampler
        Controls the sampling of windows during validation.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: Optional[int] = None,
        input_dim: int = 32,
        hidden_dim: int = 64, 
        fourier_v: int = 8,
        max_period: float = 52.18,
        bernstein_v: int = 8,
        num_layers: int = 2, 
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        distr_output: Output = StudentTOutput(),
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> None:
        default_trainer_kwargs = {
            "max_epochs": 100,
            "gradient_clip_val": 10.0,
        }
        if trainer_kwargs is not None:
            default_trainer_kwargs.update(trainer_kwargs)
        super().__init__(trainer_kwargs=default_trainer_kwargs)

        self.prediction_length = prediction_length
        self.context_length = context_length or 10 * prediction_length
        # TODO find way to enforce same defaults to network and estimator
        # somehow
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fourier_v = fourier_v
        self.max_period = max_period
        self.bernstein_v = bernstein_v
        self.num_layers = num_layers
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.distr_output = distr_output
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch

        self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=prediction_length
        )
        self.validation_sampler = validation_sampler or ValidationSplitSampler(
            min_future=prediction_length
        )

    def create_transformation(self) -> Transformation:
        return SelectFields(
            [
                FieldName.ITEM_ID,
                FieldName.INFO,
                FieldName.START,
                FieldName.TARGET,
                FieldName.FEAT_DYNAMIC_REAL
            ],
            allow_missing=True,
        ) + AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        ) 

    def create_lightning_module(self) -> pl.LightningModule:
        return BasisLightningModule(
            lr=self.lr,
            weight_decay=self.weight_decay,
            model_kwargs={
                "prediction_length": self.prediction_length,
                "context_length": self.context_length,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "fourier_v": self.fourier_v,
                "max_period": self.max_period,
                "bernstein_v": self.bernstein_v,
                "num_layers": self.num_layers,
                "distr_output": self.distr_output,
                "device": resolve_device("auto")
            },
        )

    def _create_instance_splitter(
        self, module: BasisLightningModule, mode: str
    ):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.OBSERVED_VALUES,
                FieldName.FEAT_DYNAMIC_REAL
            ],
            dummy_value=self.distr_output.value_in_support,
        )

    def create_training_data_loader(
        self,
        data: Dataset,
        module: BasisLightningModule,
        shuffle_buffer_length: Optional[int] = None,
        **kwargs,
    ) -> Iterable:
        data = Cyclic(data).stream()
        instances = self._create_instance_splitter(module, "training").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )

    def create_validation_data_loader(
        self,
        data: Dataset,
        module: BasisLightningModule,
        **kwargs,
    ) -> Iterable:
        instances = self._create_instance_splitter(module, "validation").apply(
            data, is_train=True
        )
        return as_stacked_batches(
            instances,
            batch_size=self.batch_size,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module,
    ) -> PyTorchPredictor:
        prediction_splitter = self._create_instance_splitter(module, "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=module,
            forecast_generator=self.distr_output.forecast_generator,
            batch_size=self.batch_size,
            prediction_length=self.prediction_length,
            device="auto",
        )