import torch

from basicsr.archs.rrdbnet_arch import RRDBNet

def convert_to_onnx(model_path):
    model_path = "./RealESRGAN_x4plus_anime_6B.pth"
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64)

    torch.onnx.export(
        model,
        dummy_input,
        "RealESRGAN_x4plus_anime_6B.onnx",
        opset_version=11,  # ONNX 算子集版本
        input_names=['input'],  # 输入节点名称
        output_names=['output'],  # 输出节点名称
        dynamic_axes={
            'input': {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'}
        }
    )
    print("导出 ONNX 成功！")

if __name__ == '__main__':
    model_path = "./RealESRGAN_x4plus_anime_6B.pth"



