import numpy as np
from ml_dtypes import bfloat16

from axengine import _C


class InferenceSession:
    def __init__(self, handle) -> None:
        """
        InferenceSession Collection.
        """
        self._handle = handle
        self._init_device()

    def _init_device(self) -> None:
        success = self._handle.init_device()
        if not success:
            raise SystemError("Err... Something wrong while initializing the AX System.")

    @classmethod
    def load_from_model(cls, model_path: str) -> "InferenceSession":
        """
        Load model graph to InferenceSession.

        Args:
            model_path (string): Path to model
        """
        _handle = _C.Runner()
        sess = cls(_handle)
        success = sess._handle.load_model(model_path)
        if not success:
            raise BufferError("Err... Something wrong while loading the model.")
        success = sess._handle.set_io_buffer()
        if not success:
            raise BufferError("Err... Something wrong while setting the model runtime context.")
        return sess

    def get_io_groups(self) -> int:
        return self._handle.get_io_shape_group()

    @property
    def num_io_groups(self) -> int:
        return self.get_io_groups()

    def get_active_group_id(self) -> int:
        return self._handle.get_active_group_id()

    def set_runtime_context(self, group_id: int) -> None:
        if group_id < 0 or group_id > self.num_io_groups:
            raise IndexError("Err... the id of group should be smaller than model io groups.")
        self._handle.set_io_shape_group(group_id)
        success = self._handle.set_io_buffer()
        if not success:
            raise BufferError("Err... Something wrong while setting the model runtime context.")

    def get_cmm_usage(self) -> int:
        return self._handle.get_cmm_usage()

    def get_input_names(self) -> list[str]:
        return self._handle.get_input_names()

    def get_output_names(self) -> list[str]:
        return self._handle.get_output_names()

    def get_input_shapes(self) -> list[list[int]]:
        return self._handle.get_input_shapes()

    def get_input_dtypes(self) -> list[int]:
        return self._handle.get_input_data_types()

    def get_output_shapes(self) -> list[list[int]]:
        return self._handle.get_output_shapes()

    def get_output_dtypes(self) -> list[int]:
        return self._handle.get_output_data_types()

    @property
    def input_dtypes(self):
        return self.get_input_dtypes()

    @property
    def input_shapes(self):
        return self.get_input_shapes()

    @property
    def output_dtypes(self):
        return self.get_output_dtypes()

    @property
    def output_shapes(self):
        return self.get_output_shapes()

    def feed_input_to_index(self, input_datum: np.ndarray, input_index: int) -> None:
        success = self._handle.feed_input_to_index(input_datum.tobytes(), input_index)
        if not success:
            raise BufferError(f"Err... Something wrong while reading the {input_index}th input.")

    def fetch_output_from_index(self, output_index: int) -> np.ndarray:
        data_buffer = self._handle.fetch_output_from_index(output_index).tobytes()
        data_type = self.output_dtypes[output_index]
        data_shape = self.output_shapes[output_index]

        if data_type == _C.DATA_TYPE.UINT8:
            data_array = np.frombuffer(data_buffer, np.uint8)
        elif data_type == _C.DATA_TYPE.UINT16:
            data_array = np.frombuffer(data_buffer, np.uint16)
        elif data_type == _C.DATA_TYPE.SINT8:
            data_array = np.frombuffer(data_buffer, np.int8)
        elif data_type == _C.DATA_TYPE.SINT16:
            data_array = np.frombuffer(data_buffer, np.int16)
        elif data_type == _C.DATA_TYPE.UINT32:
            data_array = np.frombuffer(data_buffer, np.uint32)
        elif data_type == _C.DATA_TYPE.SINT32:
            data_array = np.frombuffer(data_buffer, np.int32)
        elif data_type == _C.DATA_TYPE.FLOAT32:
            data_array = np.frombuffer(data_buffer, np.float32)
        elif data_type == _C.DATA_TYPE.BFLOAT16:
            data_array = np.frombuffer(data_buffer, bfloat16)
        else:
            # print("data type is fp16 maybe")
            data_array = np.frombuffer(data_buffer, np.float16)
            # warning temp support fp16
            # raise NotImplementedError(f"Currently doesn't supportes type `{data_type}' yet.")
        output_data = data_array.reshape(*data_shape)
        return output_data

    def run(self, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Returns:
            list[np.ndarray]: Output of the models.
        """
        for i, input_name in enumerate(self.get_input_names()):
            input_datum = input_feed[input_name]
            self.feed_input_to_index(input_datum, i)

        # Forward
        self._handle.forward()
        # Get outputs
        output_data = []
        for i in range(len(self.get_output_names())):
            output_data.append(self.fetch_output_from_index(i))
        return output_data
