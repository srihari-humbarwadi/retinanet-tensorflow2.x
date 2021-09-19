import os

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

import tensorrt as trt


class CalibratorBase:
    def __init__(self, image_generator, cache_file_path):
        self._logger = trt.Logger(trt.Logger.INFO)
        self._logger.min_severity = trt.Logger.Severity.VERBOSE

        self._image_generator = image_generator
        self._cache_file_path = cache_file_path

        input_spec = image_generator.get_input_spec()
        allocation_size = int(
            np.dtype(np.float32).itemsize * np.prod(input_spec))
        self._allocation = cuda.mem_alloc(allocation_size)

        self._batch_iterator = image_generator.get_batches()
        self._num_images_seen = 0

        self._logger.log(
            trt.Logger.INFO,
            'Using {} calibrator for INT8 calibration'.format(
                self.__class__.__name__))

    def get_batch_size(self):
        return self.image_generator._batch_size

    def get_batch(self):
        try:
            batch = next(self._batch_iterator)
            cuda.memcpy_htod(self._allocation, np.ascontiguousarray(batch))
            self._num_images_seen += self.image_generator._batch_size
            self._logger.log(
                trt.Logger.INFO,
                'On image {}/{}'.format(
                    self._num_images_seen,
                    self.image_generator.num_images))
            return [int(self._allocation)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file_path):
            self._logger.log(
                trt.Logger.INFO,
                'Using existing calibration cache from file: {}'.format(
                    self.cache_file_path))
            with open(self.cache_file_path, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        self._logger.log(
            trt.Logger.INFO,
            'Writing calibration cache file: {}'.format(self._cache_file_path))
        with open(self.cache_file_path, 'wb') as f:
            f.write(cache)


class IInt8EntropyCalibrator2(CalibratorBase, trt.IInt8EntropyCalibrator2):
    def __init__(self, *args, **kwargs):
        super(IInt8EntropyCalibrator2, self).__init__(*args, **kwargs)


class IInt8MinMaxCalibrator(CalibratorBase, trt.IInt8MinMaxCalibrator):
    def __init__(self, *args, **kwargs):
        super(IInt8MinMaxCalibrator, self).__init__(*args, **kwargs)


def get_calibrator(method, **kwargs):
    if 'entropy' in method:
        return IInt8EntropyCalibrator2(**kwargs)
    elif 'minmax' in method:
        return IInt8MinMaxCalibrator(**kwargs)
    else:
        raise ValueError('Invalid calibration method requested')
