import torch
from torchdiffeq import odeint
from torchdiffeq._impl.reversible_func import ReversibleODEFunc

torch.manual_seed(0)

# 1) Khởi tạo hàm thuận nghịch
state_dim = 4  # phải chẵn
func = ReversibleODEFunc(state_dim=state_dim)

# 2) Trạng thái ban đầu và thời gian
y0 = torch.randn(1, state_dim)  # batch = 1
t = torch.linspace(0., 1., 2001)  # 21 điểm từ 0 -> 1

# 3) Tích phân thuận
y_traj = odeint(func, y0, t, rtol=1e-8, atol=1e-8)
y_end = y_traj[-1]

# 4) Tích phân ngược bằng inverse_dynamics để khôi phục y0
t_rev = t.flip(0)
y_traj_rev = odeint(func, y_end, t_rev, rtol=1e-8, atol=1e-8)
y0_recovered = y_traj_rev[-1]

# 5) Đánh giá sai khác
err = (y0 - y0_recovered).abs().max().item()

print(f"y0:           {y0}")
print(f"y0_recovered: {y0_recovered}")
print(f"max |Δ|:      {err:.6f}")

assert torch.isfinite(y_traj).all()
assert torch.isfinite(y_traj_rev).all()
