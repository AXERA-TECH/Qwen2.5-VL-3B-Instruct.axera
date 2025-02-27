import numpy as np
from ml_dtypes import bfloat16
from axengine import InferenceSession


def test_input_bf16_output_bf16():

    model_path = "/root/dev/zhiqiang/model-banks/qwen/qwen2_post.axmodel"

    sess = InferenceSession.load_from_model(model_path)

    cmm_usage = sess.get_cmm_usage()
    assert cmm_usage == 148015708

    golden_data = np.load("/root/dev/zhiqiang/model-banks/qwen/post_input_output.npy", allow_pickle=True)
    input_feed = {"input": golden_data.item()["input"]}

    outputs = sess.run(input_feed)

    np.testing.assert_array_equal(outputs[0], golden_data.item()["output"])


def test_input_bf16_output_bf16_group2():

    model_path = "/root/dev/zhiqiang/model-banks/qwen/qwen2_p128_l0_together.axmodel"

    sess = InferenceSession.load_from_model(model_path)
    cmm_usage = sess.get_cmm_usage()
    assert cmm_usage == 33229032

    sess.set_runtime_context(group_id=1)
    golden_group1_data = np.load("/root/dev/zhiqiang/model-banks/qwen/group_0_l0_input_output.npy", allow_pickle=True)
    input_feed = {
        "K_cache": np.zeros((1, 1, 896), dtype=bfloat16),
        "V_cache": np.zeros((1, 1, 896), dtype=bfloat16),
        "indices": golden_group1_data.item()["indices_1"],
        "input": golden_group1_data.item()["input_1"],
        "mask": golden_group1_data.item()["mask_1"],
    }

    outputs_group1 = sess.run(input_feed)
    np.testing.assert_array_equal(outputs_group1[0], golden_group1_data.item()["K_cache_out_1"])
    np.testing.assert_array_equal(outputs_group1[1], golden_group1_data.item()["V_cache_out_1"])
    np.testing.assert_array_equal(outputs_group1[2], golden_group1_data.item()["output_1"])

    sess.set_runtime_context(group_id=0)
    golden_group0_data = np.load("/root/dev/zhiqiang/model-banks/qwen/group_1_l0_input_output.npy", allow_pickle=True)
    input_feed = {
        "K_cache": golden_group0_data.item()["K_cache"],
        "V_cache": golden_group0_data.item()["V_cache"],
        "indices": golden_group0_data.item()["indices"],
        "input": golden_group0_data.item()["input"],
        "mask": golden_group0_data.item()["mask"],
    }

    outputs_group0 = sess.run(input_feed)
    np.testing.assert_array_equal(outputs_group0[0], golden_group0_data.item()["K_cache_out"][None])
    np.testing.assert_array_equal(outputs_group0[1], golden_group0_data.item()["V_cache_out"][None])
    np.testing.assert_array_equal(outputs_group0[2], golden_group0_data.item()["output"])


if __name__ == "__main__":
    test_input_bf16_output_bf16_group2()
