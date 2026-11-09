"""Checks your loss functions in utils/kd.py against reference answers."""
import torch
import torch.nn.functional as F
from utils.kd import response_kd_loss, feature_kd_loss

torch.manual_seed(475)
results = []


def check(name, fn):
    try:
        ok = fn()
        results.append(ok)
        print(f'[{"PASS" if ok else "FAIL"}] {name}')
    except NotImplementedError:
        results.append(False)
        print(f'[TODO] {name}: not implemented yet')
    except Exception as e:
        results.append(False)
        print(f'[FAIL] {name}: {type(e).__name__}: {e}')


# ---------- response-based ----------
z_s = torch.randn(2, 21, 8, 8)
z_t = torch.randn(2, 21, 8, 8) * 3
y = torch.randint(0, 21, (2, 8, 8))
y[0, :2] = 255
valid = y != 255
a, b, tau = 0.3, 0.7, 4.0


def reference_response():
    hard = F.cross_entropy(z_s, y, ignore_index=255)
    p_t = F.softmax(z_t / tau, 1)
    kl = F.kl_div(F.log_softmax(z_s / tau, 1), p_t, reduction='none').sum(1)
    entropy = -(p_t * p_t.clamp_min(1e-12).log()).sum(1)
    return a * hard + b * (kl + entropy)[valid].mean()


check('response: matches the reference value',
      lambda: torch.allclose(response_kd_loss(z_s, z_t, y, a, b, tau), reference_response(), atol=1e-5))
check('response: alpha=1, beta=0 gives plain cross-entropy',
      lambda: torch.allclose(response_kd_loss(z_s, z_t, y, 1.0, 0.0, tau),
                             F.cross_entropy(z_s, y, ignore_index=255), atol=1e-6))


def ignored_pixels():
    z2 = z_s.clone()
    z2[0, :, :2] += 100 * torch.randn(21, 2, 8)
    return torch.allclose(response_kd_loss(z_s, z_t, y, a, b, tau),
                          response_kd_loss(z2, z_t, y, a, b, tau), atol=1e-5)


check('response: pixels labelled 255 are ignored', ignored_pixels)

# ---------- feature-based ----------
f_s = torch.randn(2, 16, 4, 4)
up = F.interpolate(f_s, size=(8, 8), mode='bilinear', align_corners=False)
f_t = torch.randn(2, 16, 8, 8)


def reference_feature():
    cos = (up * f_t).sum(1) / (up.norm(dim=1) * f_t.norm(dim=1))
    return (1 - cos).mean()


check('feature: matches the reference value',
      lambda: torch.allclose(feature_kd_loss(f_s, f_t), reference_feature(), atol=1e-5))
check('feature: identical features give 0',
      lambda: abs(feature_kd_loss(f_s, up).item()) < 1e-5)
check('feature: opposite features give 2',
      lambda: abs(feature_kd_loss(f_s, -up).item() - 2) < 1e-5)
check('feature: scaling the teacher feature does not change the loss',
      lambda: torch.allclose(feature_kd_loss(f_s, f_t), feature_kd_loss(f_s, 5 * f_t), atol=1e-5))

print(f'\n{sum(results)}/{len(results)} checks passed')
