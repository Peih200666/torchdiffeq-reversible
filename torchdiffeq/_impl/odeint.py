import torch
from torch.autograd.functional import vjp
#dopri5, rk4, tsit5, bosh3, fehlberg2 … là các phương pháp Runge–Kutta.
from .dopri5 import Dopri5Solver
from .bosh3 import Bosh3Solver
from .adaptive_heun import AdaptiveHeunSolver
from .fehlberg2 import Fehlberg2
from .fixed_grid import Euler, Midpoint, Heun2, Heun3, RK4
# fixed_grid_* và implicit_* là các phương pháp cố định bước.
from .fixed_grid_implicit import ImplicitEuler, ImplicitMidpoint, Trapezoid
from .fixed_grid_implicit import GaussLegendre4, GaussLegendre6
from .fixed_grid_implicit import RadauIIA3, RadauIIA5
from .fixed_grid_implicit import SDIRK2, TRBDF2
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .dopri8 import Dopri8Solver
from .tsit5 import Tsit5Solver
from .scipy_wrapper import ScipyWrapperODESolver
from .misc import _check_inputs, _flat_to_shape # kiểm tra tính hợp lệ của đầu vào (ví dụ: chiều, kiểu dữ liệu, hướng thời gian).
from .interp import _interp_evaluate # đưa nghiệm về lại đúng shape ban đầu (vì trong quá trình solver có thể flatten).

# create list of solvers
SOLVERS = {
    'dopri8': Dopri8Solver,
    'dopri5': Dopri5Solver,
    'tsit5': Tsit5Solver,
    'bosh3': Bosh3Solver,
    'fehlberg2': Fehlberg2,
    'adaptive_heun': AdaptiveHeunSolver,
    'euler': Euler,
    'midpoint': Midpoint,
    'heun2': Heun2,
    'heun3': Heun3,
    'rk4': RK4,
    'explicit_adams': AdamsBashforth,
    'implicit_adams': AdamsBashforthMoulton,
    'implicit_euler': ImplicitEuler,
    'implicit_midpoint': ImplicitMidpoint,
    'trapezoid': Trapezoid,
    'radauIIA3': RadauIIA3,
    'gl4': GaussLegendre4,
    'radauIIA5': RadauIIA5,
    'gl6': GaussLegendre6,
    'sdirk2': SDIRK2,
    'trbdf2': TRBDF2,
    # Backward compatibility: use the same name as before
    'fixed_adams': AdamsBashforthMoulton,
    # ~Backwards compatibility
    'scipy_solver': ScipyWrapperODESolver,
}


def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Hàm ánh xạ từ một Tensor vô hướng t (thời gian) và một Tensor y (trạng thái)
            → sang một Tensor đạo hàm dy/dt theo thời gian.
            (Ngoài ra, y cũng có thể là một tuple của nhiều Tensor.)
        y0: Tensor N-chiều (N-D) biểu diễn giá trị ban đầu của y tại thời điểm t[0].
            (Cũng có thể là một tuple các Tensor.)
        t: Tensor 1-chiều chứa chuỗi các thời điểm mà ta muốn tính nghiệm y.
            Các giá trị có thể tăng dần hoặc giảm dần.
            Phần tử đầu tiên của t được xem là thời điểm ban đầu t[0].
        rtol: Tensor float64 xác định giới hạn sai số tương đối tối đa cho mỗi phần tử của y.
            (tuy` chon)
        atol: Tensor float64 xác định giới hạn sai số tuyệt đối tối đa cho mỗi phần tử của y.
            (tuy` chon)
        method: optional string indicating the integration method to use.
        options: Từ điển (dict) cấu hình thêm cho phương pháp tích phân đã chọn.
            Chỉ có thể dùng nếu bạn đã chỉ rõ method.
        event_fn: Hàm ánh xạ trạng thái y sang một Tensor.
            Quá trình giải sẽ dừng lại khi giá trị này bằng 0.
            Nếu tham số này khác None, thì tất cả các phần tử của t trừ phần tử đầu tiên sẽ bị bỏ qua.

    Returns:
        y: Một Tensor (hoặc tuple các Tensor), trong đó chiều đầu tiên tương ứng với các thời điểm trong t.
            Mỗi phần tử dọc theo chiều đầu tiên là nghiệm của y tại thời điểm tương ứng.
            Giá trị ban đầu y0 sẽ là phần tử đầu tiên (y[0]).

    Raises:
        ValueError: if an invalid `method` is provided.
    """

    shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options) # Select Solvers

    if event_fn is None:
        solution = solver.integrate(t) #ko co solver -> giải ODE
    else:
        event_t, solution = solver.integrate_until_event(t[0], event_fn) # dừng khi event_fn(y)=0
        event_t = event_t.to(t)
        if t_is_reversed:
            event_t = -event_t
            #dùng để đồng bộ hướng thời gian của thời điểm sự kiện (event_t) với hướng tích phân (t)
            # đảm bảo rằng kết quả đúng dù bạn đang chạy xuôi hay ngược thời gian.

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes) # Khôi phục shape ban đầu -> giữ đúng cấu trúc dữ liệu của bài toán gốc

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def odeint_dense(func, y0, t0, t1, *, rtol=1e-7, atol=1e-9, method=None, options=None):

    assert torch.is_tensor(y0)  # TODO: handle tuple of tensors

    t = torch.tensor([t0, t1]).to(t0)

    shapes, func, y0, t, rtol, atol, method, options, _, _ = _check_inputs(func, y0, t, rtol, atol, method, options, None, SOLVERS)

    assert method == "dopri5" # phương pháp Dormand–Prince (bắt buộc)

    solver = Dopri5Solver(func=func, y0=y0, rtol=rtol, atol=atol, **options)    
    
    # The integration loop
    solution = torch.empty(len(t), *solver.y0.shape, dtype=solver.y0.dtype, device=solver.y0.device)
    solution[0] = solver.y0
    t = t.to(solver.dtype)
    solver._before_integrate(t)
    t0 = solver.rk_state.t0

    times = [t0]
    interp_coeffs = []

    for i in range(1, len(t)):
        next_t = t[i]
        while next_t > solver.rk_state.t1:
            solver.rk_state = solver._adaptive_step(solver.rk_state)
            t1 = solver.rk_state.t1

            if t1 != t0:
                # Step accepted.
                t0 = t1
                times.append(t1)
                interp_coeffs.append(torch.stack(solver.rk_state.interp_coeff))

        solution[i] = _interp_evaluate(solver.rk_state.interp_coeff, solver.rk_state.t0, solver.rk_state.t1, next_t)

    times = torch.stack(times).reshape(-1).cpu()
    interp_coeffs = torch.stack(interp_coeffs)

    def dense_output_fn(t_eval):
        idx = torch.searchsorted(times, t_eval, side="right")
        t0 = times[idx - 1]
        t1 = times[idx]
        coef = [interp_coeffs[idx - 1][i] for i in range(interp_coeffs.shape[1])]
        return _interp_evaluate(coef, t0, t1, t_eval)

    return dense_output_fn


def odeint_event(func, y0, t0, *, event_fn, reverse_time=False, odeint_interface=odeint, **kwargs):
    """Automatically links up the gradient from the event time."""

    if reverse_time:
        t = torch.cat([t0.reshape(-1), t0.reshape(-1).detach() - 1.0])
    else:
        t = torch.cat([t0.reshape(-1), t0.reshape(-1).detach() + 1.0])

    event_t, solution = odeint_interface(func, y0, t, event_fn=event_fn, **kwargs)

    # Dummy values for rtol, atol, method, and options.
    shapes, _func, _, t, _, _, _, _, event_fn, _ = _check_inputs(func, y0, t, 0.0, 0.0, None, None, event_fn, SOLVERS)

    if shapes is not None:
        state_t = torch.cat([s[-1].reshape(-1) for s in solution])
    else:
        state_t = solution[-1]

    # Event_fn takes in negated time value if reverse_time is True.
    if reverse_time:
        event_t = -event_t

    event_t, state_t = ImplicitFnGradientRerouting.apply(_func, event_fn, event_t, state_t)

    # Return the user expected time value.
    if reverse_time:
        event_t = -event_t

    if shapes is not None:
        state_t = _flat_to_shape(state_t, (), shapes)
        solution = tuple(torch.cat([s[:-1], s_t[None]], dim=0) for s, s_t in zip(solution, state_t))
    else:
        solution = torch.cat([solution[:-1], state_t[None]], dim=0)

    return event_t, solution


class ImplicitFnGradientRerouting(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, event_fn, event_t, state_t):
        """ event_t is the solution to event_fn """
        ctx.func = func
        ctx.event_fn = event_fn
        ctx.save_for_backward(event_t, state_t)
        return event_t.detach(), state_t.detach()

    @staticmethod
    def backward(ctx, grad_t, grad_state):
        func = ctx.func
        event_fn = ctx.event_fn
        event_t, state_t = ctx.saved_tensors

        event_t = event_t.detach().clone().requires_grad_(True)
        state_t = state_t.detach().clone().requires_grad_(True)

        f_val = func(event_t, state_t)

        with torch.enable_grad():
            c, (par_dt, dstate) = vjp(event_fn, (event_t, state_t))

        # Total derivative of event_fn wrt t evaluated at event_t.
        dcdt = par_dt + torch.sum(dstate * f_val)

        # Add the gradient from final state to final time value as if a regular odeint was called.
        grad_t = grad_t + torch.sum(grad_state * f_val)

        dstate = dstate * (-grad_t / (dcdt + 1e-12)).reshape_as(c)

        grad_state = grad_state + dstate

        return None, None, None, grad_state
