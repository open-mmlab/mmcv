# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self,
                 results: dict) -> dict | tuple[list, list] | None:

        return self.transform(results)

    @abstractmethod
    def transform(self,
                  results: dict) -> dict | tuple[list, list] | None:
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
