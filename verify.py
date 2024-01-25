import onnx

onnx_model = onnx.load('screw.onnx')

onnx.checker.check_model(onnx_model)

print('onnx模型导入成功')

print(onnx.helper.printable_graph(onnx_model.graph))