import numpy as np
from axengine import InferenceSession


def test_input_fp32_output_fp32():
    model_path = "/root/dev/zhiqiang/model-banks/flow/flow.axmodel"

    sess = InferenceSession.load_from_model(model_path)
    cmm_usage = sess.get_cmm_usage()

    assert cmm_usage == 23376643

    z_data = np.fromfile("/root/dev/zhiqiang/model-banks/flow/inputs/4/z.bin", dtype=np.float32)
    ymask_data = np.fromfile("/root/dev/zhiqiang/model-banks/flow/inputs/4/ymask.bin", dtype=np.float32)
    g_data = np.fromfile("/root/dev/zhiqiang/model-banks/flow/inputs/4/g.bin", dtype=np.float32)

    outputs = sess.run({"z": z_data, "ymask": ymask_data, "g": g_data})

    output_data = np.fromfile("/root/dev/zhiqiang/model-banks/flow/outputs/4/6797.bin", dtype=np.float32).reshape(1, 192, 120)
    np.testing.assert_array_equal(output_data, outputs[0])
