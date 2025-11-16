import warnings
import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint
from .misc import _check_inputs, _flat_to_shape, _mixed_norm, _all_callback_names, _all_adjoint_callback_names


class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):

        ctx.shapes = shapes
        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad
        ctx.event_mode = event_fn is not None

        with torch.no_grad():
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn)

            if event_fn is None:
                y = ans
                ctx.save_for_backward(t, y, *adjoint_params)
            else:
                event_t, y = ans
                ctx.save_for_backward(t, y, event_t, *adjoint_params)

        return ans

    @staticmethod
    def backward(ctx, *grad_y):
        with torch.no_grad():
            func = ctx.func
            adjoint_rtol = ctx.adjoint_rtol
            adjoint_atol = ctx.adjoint_atol
            adjoint_method = ctx.adjoint_method
            adjoint_options = ctx.adjoint_options
            t_requires_grad = ctx.t_requires_grad
            shapes = ctx.shapes 

            adjoint_options_rev = {}
            if adjoint_options is not None:
                adjoint_options_rev = {k: v for k, v in adjoint_options.items() if k != "norm"}

            # Backprop as if integrating up to event time.
            # Does NOT backpropagate through the event time.
            event_mode = ctx.event_mode
            if event_mode:
                t, y, event_t, *adjoint_params = ctx.saved_tensors
                _t = t # Lưu lại tensor t gốc 
                grad_event_t, grad_y = grad_y # bỏ qua grad của event_t 
            else:
                t, y, *adjoint_params = ctx.saved_tensors
                grad_y = grad_y[0]
                _t = t

            adjoint_params = tuple(adjoint_params)
            y_final = y[-1]

            # KIỂM TRA TÍNH THUẬN NGHỊCH 
            is_reversible = hasattr(func, "inverse_dynamics") and callable(func.inverse_dynamics)
            
            if not is_reversible:
                warnings.warn("Using standard adjoint method for non-reversible func.")

                #  HÀM AUGMENTED_DYNAMICS
                aug_state = [torch.zeros((), dtype=y.dtype, device=y.device), y[-1], grad_y[-1]]  # vjp_t, y, vjp_y
                aug_state.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params

                ##################################
                #    Set up backward ODE func    #
                ##################################
                def augmented_dynamics(t_local, y_aug):
                    # Dynamics of the original system augmented with
                    # the adjoint wrt y, and an integrator wrt t and args.
                    y = y_aug[1]
                    adj_y = y_aug[2]
                    # ignore gradients wrt time and parameters

                    with torch.enable_grad():
                        t_ = t_local.detach()
                        t_req = t_.requires_grad_(True)
                        y_req = y.detach().requires_grad_(True)

                        # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
                        # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
                        # wrt t here means we won't compute that if we don't need it.
                        func_eval = func(t_req if t_requires_grad else t_, y_req)

                        # Workaround for PyTorch bug #39784
                        _t = torch.as_strided(t_req, (), ())  # noqa
                        _y = torch.as_strided(y_req, (), ())  # noqa
                        _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)  # noqa

                        # Compute VJP
                        vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                            func_eval, (_t, _y) + adjoint_params, -adj_y,
                            allow_unused=True, retain_graph=True
                        )

                    # autograd.grad returns None if no gradient, set to zero.
                    vjp_t = torch.zeros_like(t_req) if vjp_t is None else vjp_t
                    vjp_y = torch.zeros_like(y_req) if vjp_y is None else vjp_y
                    vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                                for param, vjp_param in zip(adjoint_params, vjp_params)]

                    return (vjp_t, func_eval, vjp_y, *vjp_params)

                # Add adjoint callbacks
                for callback_name, adjoint_callback_name in zip(_all_callback_names, _all_adjoint_callback_names):
                    try:
                        callback = getattr(func, adjoint_callback_name)
                    except AttributeError:
                        pass
                    else:
                        setattr(augmented_dynamics, callback_name, callback)

                ##################################
                #       Solve adjoint ODE        #
                ##################################

                if t_requires_grad:
                    time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
                else:
                    time_vjps = None
                
                # Đi ngược từ t_N → t_0
                for i in range(len(t) - 1, 0, -1):
                    if t_requires_grad:
                        # Compute the effect of moving the current time measurement point.
                        # We don't compute this unless we need to, to save some computation.
                        func_eval = func(t[i], y[i])
                        dLd_cur_t = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                        aug_state[0] -= dLd_cur_t
                        time_vjps[i] = dLd_cur_t

                    # Tích phân ngược từ t_i → t_{i-1}
                    aug_state = odeint(
                        augmented_dynamics, tuple(aug_state),
                        t[i - 1:i + 1].flip(0),
                        rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                    )
                    aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
                    aug_state[1] = y[i - 1]  # update to use our forward-pass estimate of the state
                    aug_state[2] += grad_y[i - 1]  # update any gradients wrt state at this time point

                if t_requires_grad:
                    time_vjps[0] = aug_state[0]

                # Only compute gradient wrt initial time when in event handling mode.
                if event_mode and t_requires_grad:
                    time_vjps = torch.cat([time_vjps[0].reshape(-1), torch.zeros_like(_t[1:])])

                adj_y = aug_state[2]
                adj_params = aug_state[3:]

                return (None, None, adj_y, time_vjps, None, None, None, None, None, None, None, None, None, None, *adj_params)
            
            # KHỞI TẠO TRẠNG THÁI BAN ĐẦU CHO HỆ ADJOINT
            # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
            adj_y = grad_y[-1]  # dL/dy_N
            adj_params_integrand = tuple(torch.zeros_like(p) for p in adjoint_params)
            adj_t = torch.tensor(0., dtype=y_final.dtype, device=y_final.device)

            # Trạng thái ban đầu của hệ adjoint tại t_N
            adjoint_state = (adj_y,) + adj_params_integrand + (adj_t,)

            # ---- TÍNH LẠI QUỸ ĐẠO y(t) NGƯỢC BẰNG inverse_dynamics ---- 
            # Tích phân y ngược từ y(t_N) về y(t_0) - inverse_dynamics 
            # Lưu lại các điểm y(t_i) trên quỹ đạo ngược này
            t_rev = t.flip(0)
            y_reverse_trajectory = odeint(
                func.inverse_dynamics,
                y_final,
                t_rev,
                rtol = adjoint_rtol,
                atol = adjoint_atol,
                method = adjoint_method,
                options = adjoint_options_rev,
            )
            # y_reverse_trajectory giờ chứa [y(t_N), y(t_{N-1}), ..., y(t_0)] chính xác

            # Chuẩn bị trajectory forward
            t_fw = t # tăng dần: [t_0, ..., t_N]
            y_fw = y_reverse_trajectory.flip(0)
            f_fw = torch.stack([func(t_fw[i], y_fw[i]) for i in range(len(t_fw))], dim=0)

            # Hàm nội suy Hermite
            def hermite_interpolate(t_local):
                # tìm khoảng [t_i, t_{i+1}] sao cho t_i <= t <= t_{i+1}
                idx = torch.searchsorted(t_fw, t_local) - 1 # interval index
                idx = torch.clamp(idx, 0, len(t_fw) - 2) # bảo vệ tính an toàn của nội suy.

                t0 = t_fw[idx]
                t1 = t_fw[idx + 1]
                y0 = y_fw[idx]
                y1 = y_fw[idx + 1]
                f0 = f_fw[idx]
                f1 = f_fw[idx + 1]

                h = t1 - t0
                # tránh chia cho 0 trong trường hợp hiếm khi t trùng
                if torch.abs(h) < 1e-12:
                    return y0
                
                tau = (t_local - t0) / h

                # Hermite basis functions
                h00 = 2 * (tau ** 3) - 3 * (tau ** 2) + 1
                h10 = (tau ** 3) - 2 * (tau ** 2) + tau
                h01 = -2 * (tau ** 3) + 3 * (tau ** 2)
                h11 = (tau ** 3) - (tau ** 2)

                return h00 * y0 + h01 * y1 + h * (h10 * f0 + h11 * f1)
              
            def adjoint_dynamics(t, adjoint_state_):
                """
                Tính đạo hàm của (a(t), integral(dL/dθ), dL/dt)
                """
                adj_y_ = adjoint_state_[0] # a(t)

                # Lấy trạng thái y(t) tương ứng từ quỹ đạo ngược đã tính
                # Tìm index của t trong t.flip(0) để lấy đúng y

                # Old (error with dopri5)
                # time_index = torch.where(t.flip(0) == t)[0][0]
                # y_ = y_reverse_trajectory[time_index] # lay chinh xac y(t)

                # New - use cubic Hermite
                y_ = hermite_interpolate(t)

                # Tính f(t, y) và VJP
                with torch.enable_grad():
                    y_ = y_.detach().requires_grad_(True)

                    if t_requires_grad: # Trường hợp cần grad theo t
                        t_ = t.detach().requires_grad_(True)
                        func_eval = func(t_, y_)

                        # Tính VJP: -a^T df/dy, -a^T df/dtheta, -a^T df/dt
                        vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                            outputs=func_eval,
                            inputs=(t_, y_) + adjoint_params,
                            grad_outputs=-adj_y_,
                            retain_graph=True,
                            allow_unused=True,
                        )
                    else: # Không cần grad theo t: không đưa t vào inputs
                        t_ = t.detach()
                        func_eval = func(t_, y_)

                        # Tính VJP: -a^T df/dy, -a^T df/dtheta, -a^T df/dt
                        grads = torch.autograd.grad(
                            outputs=func_eval,
                            inputs=(y_, ) + adjoint_params,
                            grad_outputs=-adj_y_,
                            retain_graph=True,
                            allow_unused=True
                        )

                        vjp_y = grads[0]
                        vjp_params = grads[1:]
                        vjp_t = torch.zeros_like(t_)

                # Xử lý None gradients
                vjp_y = torch.zeros_like(y_) if vjp_y is None else vjp_y
                vjp_t = torch.zeros_like(t_) if vjp_t is None else vjp_t
                vjp_params = tuple(
                    torch.zeros_like(p) if vjp_p is None else vjp_p
                    for p, vjp_p in zip(adjoint_params, vjp_params)
                )

                return  (vjp_y,) + vjp_params + (vjp_t,)  # (da/dt, d(integral)/dt, d(dL/dt)/dt) 
               
            # Giải ODE Adjoint
            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype = t.dtype, device = t.device)
            else:
                time_vjps = None
            
            N = len(t)

            for i in range(N - 1, 0, -1):
                # 1) Nếu cần grad theo t, tính dL/dt_i = f(t_i, y_i) · dL/dy_i
                if t_requires_grad:
                    # y_i tương ứng với index j trên quỹ đạo nghịch:
                    # t_rev[0] = t[N-1], t_rev[N-1] = t[0] => j = N-1-i
                    j = N - 1 - i
                    y_i = y_reverse_trajectory[j]
                    func_eval_i = func(t[i], y_i)
                    dLd_t_i = func_eval_i.reshape(-1).dot(grad_y[i].reshape(-1))

                    # cập nhật adj_t an toàn
                    adjoint_state = list(adjoint_state)
                    adjoint_state[-1] = adjoint_state[-1] - dLd_t_i
                    adjoint_state = tuple(adjoint_state)

                    time_vjps[i] = dLd_t_i

                # 2) Tích phân adjoint từ t[i] về t[i-1]
                t_segment = t[i - 1:i + 1].flip(0)  # [t_i, t_{i-1}]
                adjoint_traj = odeint(
                    adjoint_dynamics,
                    adjoint_state,
                    t_segment,
                    rtol=adjoint_rtol,
                    atol=adjoint_atol,
                    method=adjoint_method,
                    options=adjoint_options_rev,
                )

                # Lấy trạng thái tại t_{i-1} (index 1)
                adjoint_state = tuple(a[1] for a in adjoint_traj)

                # 3) "Jump" do loss tại mốc t_{i-1}: adj_y(t_{i-1}) += dL/dy_{i-1}
                adj_y_prev = adjoint_state[0] + grad_y[i - 1]
                adj_params_prev = adjoint_state[1:-1]
                adj_t_prev = adjoint_state[-1]

                adjoint_state = (adj_y_prev,) + adj_params_prev + (adj_t_prev,)
            
            # Lấy kết quả cuối cùng tại t_0
            adj_y_t0 = adjoint_state[0]
            adj_params_t0 = adjoint_state[1:-1]
            adj_t_t0 = adjoint_state[-1]

            if t_requires_grad:
                time_vjps[0] = adj_t_t0
            else:
                time_vjps = None 
    
            # Trả về gradients 
            # shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol, 
            # adjoint_method, adjoint_options, t.requires_grad, *adjoint_params 
            return (None, None, adj_y_t0, time_vjps, None, None, None, None, None, None, None, None, None, None, *adj_params_t0)

def odeint_adjoint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None,
                   adjoint_rtol=None, adjoint_atol=None, adjoint_method=None, adjoint_options=None, adjoint_params=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if adjoint_params is None and not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method

    if adjoint_method != method and options is not None and adjoint_options is None:
        raise ValueError("If `adjoint_method != method` then we cannot infer `adjoint_options` from `options`. So as "
                         "`options` has been passed then `adjoint_options` must be passed as well.")

    if adjoint_options is None:
        adjoint_options = {k: v for k, v in options.items() if k != "norm"} if options is not None else {}
    else:
        # Avoid in-place modifying a user-specified dict.
        adjoint_options = adjoint_options.copy()

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)  # in case adjoint_params is a generator.

    # Filter params that don't require gradients.
    oldlen_ = len(adjoint_params)
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)
    if len(adjoint_params) != oldlen_:
        # Some params were excluded.
        # Issue a warning if a user-specified norm is specified.
        if 'norm' in adjoint_options and callable(adjoint_options['norm']):
            warnings.warn("An adjoint parameter was passed without requiring gradient. For efficiency this will be "
                          "excluded from the adjoint pass, and will not appear as a tensor in the adjoint norm.")

    # Convert to flattened state.
    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    # Handle the adjoint norm function.
    state_norm = options.get("norm", None)
    handle_adjoint_norm_(adjoint_options, shapes, state_norm)

    ans = OdeintAdjointMethod.apply(shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol,
                                    adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)

    if event_fn is None:
        solution = ans
    else:
        event_t, solution = ans
        event_t = event_t.to(t)
        if decreasing_time:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def handle_adjoint_norm_(adjoint_options, shapes, state_norm):
    """In-place modifies the adjoint options to choose or wrap the norm function."""

    # This is the default adjoint norm on the backward pass: a mixed norm over the tuple of inputs.
    def default_adjoint_norm(tensor_tuple):
        t, y, adj_y, *adj_params = tensor_tuple
        # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
        return max(t.abs(), state_norm(y), state_norm(adj_y), _mixed_norm(adj_params))

    if "norm" not in adjoint_options:
        # `adjoint_options` was not explicitly specified by the user. Use the default norm.
        adjoint_options["norm"] = default_adjoint_norm
    else:
        # `adjoint_options` was explicitly specified by the user...
        try:
            adjoint_norm = adjoint_options['norm']
        except KeyError:
            # ...but they did not specify the norm argument. Back to plan A: use the default norm.
            adjoint_options['norm'] = default_adjoint_norm
        else:
            # ...and they did specify the norm argument.
            if adjoint_norm == 'seminorm':
                # They told us they want to use seminorms. Slight modification to plan A: use the default norm,
                # but ignore the parameter state
                def adjoint_seminorm(tensor_tuple):
                    t, y, adj_y, *adj_params = tensor_tuple
                    # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
                    return max(t.abs(), state_norm(y), state_norm(adj_y))
                adjoint_options['norm'] = adjoint_seminorm
            else:
                # And they're using their own custom norm.
                if shapes is None:
                    # The state on the forward pass was a tensor, not a tuple. We don't need to do anything, they're
                    # already going to get given the full adjoint state as (t, y, adj_y, adj_params)
                    pass  # this branch included for clarity
                else:
                    # This is the bit that is tuple/tensor abstraction-breaking, because the odeint machinery
                    # doesn't know about the tupled nature of the forward state. We need to tell the user's adjoint
                    # norm about that ourselves.

                    def _adjoint_norm(tensor_tuple):
                        t, y, adj_y, *adj_params = tensor_tuple
                        y = _flat_to_shape(y, (), shapes)
                        adj_y = _flat_to_shape(adj_y, (), shapes)
                        return adjoint_norm((t, *y, *adj_y, *adj_params))
                    adjoint_options['norm'] = _adjoint_norm
