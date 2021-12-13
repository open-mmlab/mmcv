# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict


class BaseTransform(metaclass=ABCMeta):

    def __call__(self, results: Dict) -> Dict:

        return self.transform(results)

    @abstractmethod
    def transform(self, results: Dict) -> Dict:
        pass
