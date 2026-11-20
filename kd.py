"""
q1_distillation/kd.py   (Q1 part 3: the knowledge distillation framework)
-------------------------------------------------------------------------
Implements the four required pieces, combined into one loss:

  1. Soft-target distillation : the student learns to match the teacher's whole probability
                                distribution, not only the correct class.
  2. Temperature softmax      : softmax(z / T). T > 1 "softens" the distribution so the small
                                probabilities (e.g. "this cat looks a bit like a dog") become
                                visible - this is the "dark knowledge" the teacher transfers.
  3. Hard + soft labels       : loss = (1 - alpha) * CE(student, true label)
                                       + alpha * T^2 * KL(teacher_T || student_T)
                                The T^2 factor keeps the soft-loss gradients at the same scale as
                                the hard-loss gradients when T changes (Hinton et al., 2015).
  4. Feature distillation     : (FitNets, Romero et al., 2015) the student's intermediate 4x4
                                feature map is mapped by a learned 1x1-conv "adapter" to the
                                teacher's 1024 channels and pulled towards the teacher's
                                feature map with a (scale-normalised) mean-squared error.

Total loss = (1 - alpha) * L_hard + alpha * T^2 * L_soft + beta * L_feature
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_t(logits, T):
    """Temperature-based softmax."""
    return F.softmax(logits / T, dim=1)


def soft_target_loss(student_logits, teacher_logits, T):
    """KL(teacher_T || student_T) * T^2   (soft-target knowledge distillation)."""
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = softmax_t(teacher_logits, T)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * T * T


class FeatureAdapter(nn.Module):
    """1x1 conv + BN that maps the student's hint (C_s channels) to the teacher's (C_t channels).
    If the spatial sizes differ, the student map is resized to the teacher's."""

    def __init__(self, c_student, c_teacher):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c_student, c_teacher, 1, bias=False), nn.BatchNorm2d(c_teacher))

    def forward(self, s, size):
        if s.shape[-2:] != size:
            s = F.adaptive_avg_pool2d(s, size)
        return self.proj(s)


def feature_loss(student_hint, teacher_hint, adapter):
    """Mean-squared error between adapted student features and teacher features, divided by the
    teacher features' mean square, so the term is ~1 at the start whatever the feature scale."""
    s = adapter(student_hint, teacher_hint.shape[-2:])
    return F.mse_loss(s, teacher_hint) / (teacher_hint.pow(2).mean() + 1e-8)


class DistillationLoss(nn.Module):
    """Combined hard-label + soft-label + feature distillation loss."""

    def __init__(self, c_student, c_teacher, T=4.0, alpha=0.7, beta=1.0):
        super().__init__()
        self.T, self.alpha, self.beta = T, alpha, beta
        self.adapter = FeatureAdapter(c_student, c_teacher) if beta > 0 else None

    def forward(self, s_logits, s_hint, t_logits, t_hint, labels):
        hard = F.cross_entropy(s_logits, labels)
        soft = soft_target_loss(s_logits, t_logits, self.T)
        feat = feature_loss(s_hint, t_hint, self.adapter) if self.beta > 0 else torch.zeros(())
        total = (1 - self.alpha) * hard + self.alpha * soft + self.beta * feat
        return total, {"hard": hard.item(), "soft": soft.item(), "feat": feat.item()}
