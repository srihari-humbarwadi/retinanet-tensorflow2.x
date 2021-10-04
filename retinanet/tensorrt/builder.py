import tensorrt as trt


class TensorRTBuilder:

    def __init__(
            self,
            onnx_path='model.onnx',
            engine_path='model.trt',
            workspace=1,
            precision='fp32',
            calibrator=None,
            dla_core=None,
            debug=False):

        self._logger = trt.Logger(trt.Logger.INFO)
        if debug:
            self._logger.log(trt.Logger.INFO, 'Logging VERBOSELY')
            self._logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self._logger, namespace='')

        self._engine_path = engine_path
        self._onnx_path = onnx_path
        self._calibrator = calibrator
        self._dla_core = dla_core

        self._precision = precision
        self._workspace = int((1 << 30) * workspace)

        self._builder = trt.Builder(self._logger)
        self._config = self._builder.create_builder_config()
        self._config.max_workspace_size = self._workspace
        self._build_network()

    def _build_network(self):
        self._logger.log(trt.Logger.INFO, 'Building network from ONNX model')

        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self._network = self._builder.create_network(network_flags)
        self._onnx_parser = trt.OnnxParser(self._network, self._logger)

        with open(self._onnx_path, 'rb') as f:
            self._onnx_parser.parse(f.read())

    def build_engine(self):
        if self._precision in {'fp16', 'int8'}:
            if self._builder.platform_has_fast_fp16:
                self._logger.log(
                    trt.Logger.INFO,
                    'Platform has native support for FP16')
            else:
                self._logger.log(
                    trt.Logger.WARNING,
                    'Platform has no native support for FP16, this will impact '
                    'performace adversely')
            self._config.set_flag(trt.BuilderFlag.FP16)

            if self._precision == 'int8':
                if self._builder.platform_has_fast_int8:
                    self._logger.log(
                        trt.Logger.INFO,
                        'Platform has native support for INT8')
                else:
                    self._logger.log(
                        trt.Logger.WARNING,
                        'Platform has no native support for INT8, this will impact '
                        'performace adversely')
                self._config.set_flag(trt.BuilderFlag.INT8)
                self._config.int8_calibrator = self._calibrator

        if self._dla_core is not None:
            self._config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            self._config.default_device_type = trt.DeviceType.DLA
            self._config.DLA_core = self._dla_core

            self._logger.log(
                trt.Logger.WARNING,
                'Using DLA Core: {} for engine'.format(self._dla_core))

        serialized_engine = self._builder.build_serialized_network(
            network=self._network,
            config=self._config)

        with open(self._engine_path, 'wb') as f:
            f.write(serialized_engine)

        self._logger.log(
            trt.Logger.INFO,
            'TensorRT Engine dumped at: {}'.format(self._engine_path))
