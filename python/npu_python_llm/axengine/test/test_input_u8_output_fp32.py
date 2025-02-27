import numpy as np
from axengine import InferenceSession


def test_input_u8_output_fp32():

    model_path = "/root/dev/zhiqiang/model-banks/mobilenetv2/mobilenetv2.axmodel"

    sess = InferenceSession.load_from_model(model_path)

    cmm_usage = sess.get_cmm_usage()
    assert cmm_usage == 4421724
    image_path = "/root/dev/zhiqiang/model-banks/mobilenetv2/inputs/0/input.bin"
    input_data = np.fromfile(image_path, dtype=np.uint8)

    # Inference
    outputs = sess.run({"input": input_data})

    output_data = np.fromfile("/root/dev/zhiqiang/model-banks/mobilenetv2/outputs/0/output.bin", dtype=np.float32).reshape(1, 1000)
    np.testing.assert_array_equal(output_data, outputs[0])


def _test_input_u8_output_fp32_group3():

    model_path = "/root/dev/zhiqiang/model-banks/mobilenetv2/mobilenetv2-group3.axmodel"

    sess = InferenceSession.load_from_model(model_path)

    cmm_usage = sess.get_cmm_usage()
    assert cmm_usage == 16423856
    image_path = "/root/dev/zhiqiang/model-banks/mobilenetv2/inputs/0/input.bin"
    input_data = np.fromfile(image_path, dtype=np.uint8)

    # Inference
    outputs = sess.run({"data": input_data})
    print(f"{sess._handle.get_io_shape_group() = }")
    print(f"{sess.get_input_names() = }")

    # output_data = np.fromfile("/root/dev/zhiqiang/model-banks/mobilenetv2/outputs/0/output.bin", dtype=np.float32).reshape(1, 1000, 1, 1)
    # np.testing.assert_array_equal(output_data, outputs[0])


if __name__ == "__main__":
    _test_input_u8_output_fp32_group3()
