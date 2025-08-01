
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import onnx
import sys
# onnx_output = "Qwen2.5-VL-3B-Instruct_vision.onnx"
onnx_output = sys.argv[1]
onnx_model = onnx.load(onnx_output)
print("IR 版本:", onnx_model.ir_version)
print("操作集:", onnx_model.opset_import)
# convert model
model_simp, check = onnxsim.simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, onnx_output, 
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=onnx_output+".data",)
print("onnx simpilfy successed, and model saved in {}".format(onnx_output))