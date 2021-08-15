import numpy as np

import mmcv


class RandomFlip:
    """Random Flip. If the input dict contains the key "flip", then the flag
    will be used, otherwise it will be randomly decided by a ratio specified in
    the init method.

    When random flip is enabled, ``prob``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:
    - ``prob`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``prob`` .
        E.g., ``prob=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``prob`` is float, ``direction`` is list of string: the image wil
        be ``direction[i]``ly flipped with probability of
        ``prob/len(direction)``.
        E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``prob`` is list of float, ``direction`` is list of string:
        given ``len(prob) == len(direction)``, the image wil
        be ``direction[i]``ly flipped with probability of ``prob[i]``.
        E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
        of 0.3, vertically with probability of 0.5.

    Args:
        prob (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``prob``. Each
            element in ``prob`` indicates the flip probability of
            corresponding direction. Default: 'horizontal'.
        flip_fields (str | list[str] | None, optional): The name list of data
            fields that need flip. If given ``None``, we adopt all of the
            pre-defined names in ``_flip_functions_map``. If the name is not
            in ``_flip_functions_map``, we will adopt the `img_flip` function
            to flip the data. Default: None.
    """

    # `_flip_functions_map` contain the map between the field name (key) and
    # the name of flip function (value).
    # If a flied name is not in this map, we will directly use `img_flip`
    # flip function.
    _flip_functions_map = dict(
        img='img_flip', bbox='bbox_flip', seg='img_flip', mask='img_flip')

    def __init__(self, prob=None, direction='horizontal', flip_fields=None):
        if isinstance(prob, list):
            assert mmcv.is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        elif prob is None:
            pass
        else:
            raise ValueError('probs must be None, float, ' 'or list of float')

        self.prob = prob

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(prob, list):
            assert len(self.prob) == len(self.direction)

        # set flip_fields
        self.flip_fields = flip_fields or list(self._flip_functions_map.keys())

        # set direction list with None
        if isinstance(self.direction, list):
            # None means non-flip
            self.direction_list = self.direction + [None]
        else:
            # None means non-flip
            self.direction_list = [self.direction, None]

        # set prob list with non_flip_prob
        if isinstance(self.prob, list):
            non_flip_prob = 1 - sum(self.prob)
            self.prob_list = self.prob + [non_flip_prob]
        elif self.prob is not None:
            non_flip_prob = 1 - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(self.direction_list) - 1)
            self.prob_list = [single_ratio] * (len(self.direction_list) -
                                               1) + [non_flip_prob]

    def bbox_flip(self, results, key, direction):
        """Flip bboxes.

        Args:
            results (dict): Result dict from loading pipeline.
            key (str): Key for the necessary data.
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """

        bboxes = results[key]
        img_shape = results['img_shape']

        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        # direction == 'diagonal'
        else:
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]

        return flipped

    def img_flip(self, results, key, direction):
        """Flip an image.
        Args:
            results (dict): Result dict from loading pipeline.
            key (str): Key for the necessary data.
            direction (str): The flip direction, either "horizontal" or
                "vertical" or "diagonal".

        Returns:
            ndarray: The flipped image.
        """
        img = results[key]
        # use copy() to make numpy stride positive
        return mmcv.imflip(img, direction).copy()

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            cur_dir = np.random.choice(self.direction_list, p=self.prob_list)

            results['flip'] = cur_dir is not None

        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir

        if results['flip']:
            for field in self.flip_fields:
                # get flip function for the `field`
                if field in self._flip_functions_map:
                    flip_func = getattr(self, self._flip_functions_map[field])
                else:
                    flip_func = self.img_flip

                # a data field may contain a list of keys
                for key in results.get(field + '_fields', []):
                    results[key] = flip_func(results, key,
                                             results['flip_direction'])

                # the field name may be the key of the data
                if field in results and f'{field}_fields' not in results:
                    results[field] = flip_func(results, field,
                                               results['flip_direction'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'
