# Start

```bash
pip install -e .
```

# Test

```bash
pytest test
```

# Usage

```python
import numpy as np
from axengine import InferenceSession

model_path = "/root/test/compiled_u16.axmodel"

# Initialization
sess = InferenceSession.load_from_model(model_path)

# Basic verification
cmm_usage = sess.get_cmm_usage()
assert cmm_usage == 214708068

# Prepare input data
input_path0 = "/root/test/inputs/onnx__ReduceMean_0.bin"
input_data0 = np.fromfile(input_path0, dtype=np.float32)
input_path1 = "/root/test/inputs/onnx__ReduceMean_1.bin"
input_data1 = np.fromfile(input_path1, dtype=np.float32)
input_feed = {"onnx::ReduceMean_0": input_data0, "onnx::ReduceMean_1": input_data1}

# Inference
outputs = sess.run(input_feed)

# Verify the inference outputs
output_data0 = np.fromfile("/root/test/outputs/inp_3.bin", dtype=np.float32).reshape(1, 4, 4, 2048, 336)
output_data1 = np.fromfile("/root/test/outputs/inp.bin", dtype=np.float32).reshape(1, 8, 343980)
np.testing.assert_allclose(output_data0, outputs["inp.3"])
np.testing.assert_allclose(output_data1, outputs["inp"])
```
