import torch
import torch.nn.functional as F


def response_kd_loss(z_s, z_t, y, alpha=0.5, beta=0.5, tau=4.0, ignore_index=255):
    """Response-based KD loss.
    z_s, z_t: (N, C, H, W) student and teacher logits. y: (N, H, W) labels."""
    # Term 1: ordinary cross-entropy with the ground-truth labels (T = 1)
    hard = F.cross_entropy(z_s, y, ignore_index=ignore_index)

    # Term 2: cross-entropy between the softened teacher and student distributions (T = tau)
    p_t = F.softmax(z_t / tau, dim=1)              # (N, C, H, W) teacher probabilities
    log_p_s = F.log_softmax(z_s / tau, dim=1)      # (N, C, H, W) student log-probabilities
    soft_per_pixel = -(p_t * log_p_s).sum(dim=1)   # (N, H, W): -sum over the 21 classes

    valid = y != ignore_index                      # drop the 255 border pixels
    soft = soft_per_pixel[valid].mean()

    return alpha * hard + beta * soft


def feature_kd_loss(f_s, f_t):
    """Feature-based KD loss: 1 - cosine similarity, averaged over all positions.
    f_s: (N, C, h, w) projected student features. f_t: (N, C, H, W) teacher features."""
    # Match the teacher's spatial size (student 'high' is stride 32, teacher layer4 is stride 8)
    f_s = F.interpolate(f_s, size=f_t.shape[-2:], mode='bilinear', align_corners=False)
    # Cosine similarity across channels at every pixel -> (N, H, W), values in [-1, 1]
    cos = F.cosine_similarity(f_s, f_t, dim=1)
    return (1 - cos).mean()
