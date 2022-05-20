# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional


class BaseTransform(metaclass=ABCMeta):

    def __call__(self, results: Dict) -> Optional[Dict]:

        return self.transform(results)

    @abstractmethod
    def transform(self, results: Dict) -> Optional[Dict]:
        """The transform function. All subclass of BaseTransform should
        override this method.

        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
