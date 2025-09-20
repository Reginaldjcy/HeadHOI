# 1.引用模型
# 2.加载预训练权重
# 3.前向传播测试
# 4.加入数据集
import os
import argparse
import torch
import torchvision
import torch.distributed as dist
from deep_head_pose.code.hopenet import Hopenet


def build_model(args):
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    if os.path.exists(args.pretrained):
        if dist.is_initialized():
            print(f"Rank {dist.get_rank()}: Load weights for the object detector from {args.pretrained}")
        else:
            print(f"Load weights for the object detector from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default='checkpoints/hopenet_robust_alpha1.pkl', help='Path to a pretrained detector')
    args = parser.parse_args()

    model = build_model(args)

    x = torch.randn(1, 3, 224, 224)  # 假设输入是 224x224
    with torch.no_grad():
        yaw, pitch, roll = model(x)
    print(yaw.shape, pitch.shape, roll.shape)