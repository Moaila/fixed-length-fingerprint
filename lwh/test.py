import torch

pre = "./models/DeepPrint_TexMinu_512/best_model.pyt"
fin = "./examples/out/finetune_scheme1_gmfs/best_model_finetuned.pyt"

def get_sd(p):
    ck = torch.load(p, map_location="cpu")
    if isinstance(ck, dict) and "model_state_dict" in ck:
        return ck["model_state_dict"]
    return ck

sd0 = get_sd(pre)
sd1 = get_sd(fin)

k0 = set(sd0.keys())
k1 = set(sd1.keys())

print("pre keys:", len(k0))
print("fin keys:", len(k1))
print("missing in fin (pre - fin):", len(k0-k1))
print("extra in fin (fin - pre):", len(k1-k0))

# 再看 shape 不匹配的情况
mismatch = []
for k in (k0 & k1):
    if tuple(sd0[k].shape) != tuple(sd1[k].shape):
        mismatch.append((k, tuple(sd0[k].shape), tuple(sd1[k].shape)))
print("shape mismatch:", len(mismatch))
print("example mismatches:", mismatch[:10])
