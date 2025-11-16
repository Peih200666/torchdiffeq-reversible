import torch
from torch import nn
from torchdiffeq import odeint
from torchdiffeq._impl.adjoint import odeint_adjoint
from torchdiffeq._impl.reversible_func import ReversibleODEFunc

#  Hàm dựng Reversible ODE func
def make_reversible_func(state_dim):
    # state_dim = kích thước cuối cùng của y
    return ReversibleODEFunc(state_dim)

#  Loss với autograd trực tiếp
#  (odeint tiêu chuẩn)
def loss_direct(y0, t):
    func = make_reversible_func(y0.shape[-1])
    y = odeint(func, y0, t)
    return (y ** 2).sum()

#  Loss với odeint_adjoint (reversible adjoint)
def loss_adjoint(y0, t):
    func = make_reversible_func(y0.shape[-1])
    y = odeint_adjoint(func, y0, t)
    return (y ** 2).sum()

#  Test: gradient autograd vs adjoint
def test_gradient_match():
    # y0 2D: [batch, state_dim]
    y0 = torch.tensor([[1.0, 2.0]], requires_grad=True)
    t = torch.linspace(0., 1., 20)

    # ground truth (autograd qua odeint thường)
    loss_gt = loss_direct(y0, t)
    grad_gt = torch.autograd.grad(loss_gt, y0)[0]

    # adjoint
    y0_adj = y0.clone().detach().requires_grad_(True)
    loss_ad = loss_adjoint(y0_adj, t)
    grad_ad = torch.autograd.grad(loss_ad, y0_adj)[0]

    print("Ground truth grad:", grad_gt)
    print("Adjoint grad     :", grad_ad)
    print("Diff max         :", (grad_gt - grad_ad).abs().max().item())

if __name__ == "__main__":
    test_gradient_match()