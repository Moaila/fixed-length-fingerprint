import torch

src = r"D:\AAAYan\ZhiWen\FixLength\fixed-length-fingerprint\finetuned_sd4_metric\best_arcface.pth"
dst = r"D:\AAAYan\ZhiWen\FixLength\fixed-length-fingerprint\finetuned_sd4_metric\best_model.pyt"

ckpt = torch.load(src, map_location="cpu")

torch.save(
    {"model_state_dict": ckpt["model_state_dict"]},
    dst
)

print("saved ->", dst)
