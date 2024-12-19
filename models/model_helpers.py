import json
import os
from typing import Union, List

from diffusers.models.modeling_utils import ModelMixin

from utils import load_model


class TemporalModelMixin(ModelMixin):
    def __init__(self):
        super().__init__()

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, subfolder: str = None, config_type: str = "video",
                        pretrained_type: str = None, keywords: Union[str, List[str]] = None):
        """

        Args:
            pretrained_model_path: Path to the local pretrained model
            subfolder:
            config_type:
            pretrained_type:
            keywords:

        Returns:

        """
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        else:
            subfolder = pretrained_model_path.split(os.sep)[-1]

        if pretrained_type is None:
            pretrained_type = config_type

        config_file = os.path.join(pretrained_model_path, f'{subfolder}_{config_type}_config.json')

        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__

        model = cls.from_config(config)

        model_file = os.path.join(pretrained_model_path, f"{subfolder}_{pretrained_type}.bin")

        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        load_model(model, model_file, device="cuda", keywords=keywords)

        return model
