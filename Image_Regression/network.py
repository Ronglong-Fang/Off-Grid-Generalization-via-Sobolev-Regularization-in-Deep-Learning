"""
Multi-grade network for 2D image regression.

Input: a coordinate x in [-1,1]^2.  Output: 3 colour channels.
Grade k is a small network g_k; grade 1 reads x, grade k >= 2 reads
the hidden features of grade k-1.  The model after k grades is

    F_k = F_{k-1} + eps_k g_k,

where eps_k is a fixed constant set by TrainingApp to the RMS training
residual of F_{k-1}.  The constants are
not trained; they are stored in the `grade_scales` buffer so that
checkpoints reproduce them.
"""

import torch
import torch.nn as nn


class SingleGradeBlock(nn.Module):
    """n_layers hidden Linear + GELU layers."""

    def __init__(self, in_dim, width, n_layers=3):
        super().__init__()
        layers = []
        d = int(in_dim)
        for _ in range(int(n_layers)):
            layers.append(nn.Linear(d, width))
            layers.append(nn.GELU())
            d = int(width)
        self.block = nn.Sequential(*layers)

    def forward(self, z):
        return self.block(z)


class SingleGradeNet(nn.Module):
    def __init__(self, in_dim, width, out_dim, n_layers=3):
        super().__init__()
        self.feature_block = SingleGradeBlock(in_dim, width, n_layers=n_layers)
        self.output_layer = nn.Linear(width, out_dim)

    def forward(self, x):
        features = self.feature_block(x)
        return self.output_layer(features), features


class FixedDepthMGDL2D(nn.Module):
    def __init__(self, width, max_grades=4, n_layers=3, in_dim=2, out_dim=3):
        super().__init__()
        self.width = int(width)
        self.max_grades = int(max_grades)
        self.n_layers = int(n_layers)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.grades = nn.ModuleList()
        for idx in range(self.max_grades):
            d_in = self.in_dim if idx == 0 else self.width
            self.grades.append(
                SingleGradeNet(in_dim=d_in, width=self.width,
                               out_dim=self.out_dim, n_layers=self.n_layers)
            )

        # Per-grade scales eps_k of F_k = F_{k-1} + eps_k g_k; fixed, not trained.
        self.register_buffer("grade_scales", torch.ones(self.max_grades))

    def set_grade_scale(self, grade, scale):
        """Set the constant eps_k for grade k (1-based).

        TrainingApp calls this once at the start of grade k, with
        scale = the RMS training residual of F_{k-1}.  The value is
        fixed for the rest of training and saved with the model.
        """
        if not (1 <= grade <= self.max_grades):
            raise ValueError(
                f"grade must be between 1 and {self.max_grades}")
        with torch.no_grad():
            self.grade_scales[grade - 1] = float(scale)

    def set_trainable_grade(self, active_grade):
        if not (1 <= active_grade <= self.max_grades):
            raise ValueError(
                f"active_grade must be between 1 and {self.max_grades}"
            )
        for param in self.parameters():
            param.requires_grad = False
        for param in self.grades[active_grade - 1].parameters():
            param.requires_grad = True

    def forward(self, x, current_grade):
        """Evaluate F_k(x) = sum_{i=1}^{k} eps_i g_i(x) for k = current_grade,
        where eps_i = grade_scales[i-1] is the constant set by
        set_grade_scale.

        The frozen grades 1..k-1 are evaluated inside the autograd graph
        (no torch.no_grad()).  Their parameters have requires_grad=False,
        so only grade k receives gradients.  Keeping them in the graph is
        necessary because the off-grid penalty is applied to F_k, not to
        g_k alone: the derivatives of F_k with respect to x pass through
        every grade.
        """
        if not (1 <= current_grade <= self.max_grades):
            raise ValueError(
                f"current_grade must be between 1 and {self.max_grades}"
            )

        out, feat = self.grades[0](x)
        total = self.grade_scales[0] * out
        for i in range(1, current_grade):
            out, feat = self.grades[i](feat)
            total = total + self.grade_scales[i] * out
        return total
