import os
import torch
import logging

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from transformers import pipeline
from label_studio_sdk.objects import PredictionValue
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


class ZeroShotClassifier(LabelStudioMLBase):
    """
    BERT-based text classification model for Label Studio

    This model uses the Hugging Face Transformers library to fine-tune a BERT model for text classification.
    Use any model for [AutoModelForSequenceClassification](https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#automodelforsequenceclassification)
    The model is trained on the labeled data from Label Studio and then used to make predictions on new data.

    Parameters:
    -----------
    baseline_model_name : str
        The name of the baseline model to use for training
    """

    baseline_model_name = os.getenv(
        "BASELINE_MODEL_NAME", "MoritzLaurer/bge-m3-zeroshot-v2.0"
    )
    _model = None

    def get_labels(self):
        li = self.label_interface
        from_name, _, _ = li.get_first_tag_occurence("Choices", "Text")
        tag = li.get_tag(from_name)
        return tag.labels

    def setup(self):
        self.set("model_version", f"{self.__class__.__name__}-v0.0.1")

    def _lazy_init(self):
        if not self._model:
            self._model = pipeline(
                "zero-shot-classification", model=self.baseline_model_name, device=0
            )

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
        :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """

        # TODO: this may result in single-time timeout for large models - consider adjusting the timeout on Label Studio side
        self._lazy_init()

        li = self.label_interface
        from_name, to_name, value = li.get_first_tag_occurence("Choices", "Text")
        texts = [self.preload_task_data(task, task["data"][value]) for task in tasks]
        classes_verbalized = self.get_labels()
        model_predictions = self._model(texts[0], classes_verbalized, multi_label=False)
        predictions = []

        threshold = 0.06
        scores = model_predictions["scores"]
        scores_above_threshold = [score for score in scores if score > threshold]
        count_above_threshold = len(scores_above_threshold)

        if count_above_threshold > 0:
            region = li.get_tag(from_name).label(
                model_predictions["labels"][:count_above_threshold]
            )
            average_confidence = sum(scores_above_threshold) / count_above_threshold
            pv = PredictionValue(
                score=average_confidence,
                result=[region],
                model_version=self.get("model_version"),
            )
            predictions.append(pv)
            return ModelResponse(predictions=predictions)
