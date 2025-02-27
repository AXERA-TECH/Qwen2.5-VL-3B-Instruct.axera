import numpy as np
from axengine import InferenceSession


def test_input_s32_output_fp32():
    model_path = "/root/dev/zhiqiang/model-banks/whisper/small-decoder-main.axmodel"

    sess = InferenceSession.load_from_model(model_path)
    cmm_usage = sess.get_cmm_usage()

    assert cmm_usage == 293791830

    WHISPER_SOT = 50258
    WHISPER_TRANSCRIBE = 50359
    WHISPER_NO_TIMESTAMPS = 50363

    SOT_SEQUENCE = np.array([WHISPER_SOT, 50260, WHISPER_TRANSCRIBE, WHISPER_NO_TIMESTAMPS], dtype=np.int32)
    n_layer_cross_k = np.fromfile("/root/dev/zhiqiang/model-banks/whisper/n_layer_cross_k.bin", dtype=np.float32)
    n_layer_cross_v = np.fromfile("/root/dev/zhiqiang/model-banks/whisper/n_layer_cross_v.bin", dtype=np.float32)

    logits, n_layer_self_k_cache, n_layer_self_v_cache = sess.run(input_feed={
        "tokens": SOT_SEQUENCE,
        "n_layer_cross_k": n_layer_cross_k,
        "n_layer_cross_v": n_layer_cross_v
    })
