import torch
import torch.nn.functional as F

def response_kd_loss(z_s, z_t, y, alpha, beta, tau, ignore_index=255):
    first = F.cross_entropy(z_s, y, ignore_index=ignore_index)
    p_t = F.softmax(z_t / tau, dim=1)
    logp_s = F.log_softmax(z_s / tau, dim=1)
    per_pixel = -(p_t * logp_s).sum(dim=1)      # (N, H, W)
    second = per_pixel[y != ignore_index].mean()
    return alpha * first + beta * second


def feature_kd_loss(f_s, f_t):
    f_s = F.interpolate(f_s, size=f_t.shape[-2:], mode="bilinear", align_corners=False)
    return (1 - F.cosine_similarity(f_s, f_t, dim=1)).mean()


