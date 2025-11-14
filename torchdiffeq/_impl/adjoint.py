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

            # Backprop as if integrating up to event time.
            # Does NOT backpropagate through the event time.
            event_mode = ctx.event_mode
            if event_mode:
                t, y, event_t, *adjoint_params = ctx.saved_tensors
                _t = t # Lưu lại tensor t gốc 
                t = torch.cat([t[0].reshape(-1), event_t.reshape(-1)])
                grad_y = grad_y[1] # Lấy grad của y, bỏ qua grad của event_t 
            else:
                t, y, *adjoint_params = ctx.saved_tensors
                grad_y = grad_y[0]

            adjoint_params = tuple(adjoint_params)
            y_final = y[-1]

            # KIỂM TRA TÍNH THUẬN NGHỊCH 
            is_reversible = hasattr(func, "inverse_dynamics") and callable(func.inverse_dynamics)

            if not is_reversible:
                warnings.warn("Using standard adjoint method for non-reversible func.")
                
            
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
            y_reverse_trajectory = odeint(
                func.inverse_dynamics,
                y_final,
                t.flip(0),
                rtol = adjoint_rtol,
                atol = adjoint_atol,
                method = adjoint_method,
                options = adjoint_options
            )
            # y_reverse_trajectory giờ chứa [y(t_N), y(t_{N-1}), ..., y(t_0)] chính xác

            ##################################
            #    Set up backward ODE func    #
            ##################################

            # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
            def adjoint_dynamics(t, adjoint_state_):
                """
                Tính đạo hàm của (a(t), integral(dL/dθ), dL/dt)
                """
                adj_y_ = adjoint_state_[0] # a(t)

                # Lấy trạng thái y(t) tương ứng từ quỹ đạo ngược đã tính
                # Tìm index của t trong t.flip(0) để lấy đúng y
                time_index = torch.where(t.flip(0) == t)[0][0]
                y_ = y_reverse_trajectory[time_index] # lay chinh xac y(t)

                # Tính f(t, y) và VJP
                with torch.enable_grad():
                    t_ = t.detach().requires_grad_(True)
                    y_ = y_.detach().requires_grad_(True)

                     # Tính f(t, y) cho VJP 
                    func_eval = func(t_, y_)

                    # Tính VJP: -a^T df/dy, -a^T df/dtheta, -a^T df/dt
                    vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                        outputs=func_eval,
                        inputs=(t_, y_) + adjoint_params,
                        grad_outputs=-adj_y_,
                        retain_graph=True,
                        allow_unused=True
                    )

                # Xử lý None gradients
                vjp_y = torch.zeros_like(y_) if vjp_y is None else vjp_y
                vjp_t = torch.zeros_like(t_) if vjp_t is None else vjp_t
                vjp_params = tuple(
                    torch.zeros_like(p) if vjp_p is None else vjp_p
                    for p, vjp_p in zip(adjoint_params, vjp_params)
                )

                return (vjp_y,) + vjp_params + (vjp_t,)  # (da/dt, d(integral)/dt, d(dL/dt)/dt) 
            
            # Giải ODE Adjoint
            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype = t.dtype, device = t.device)
            
            # Tích phân hệ adjoint ngược thời gian từ t_N về t_0
            adjoint_outputs = odeint(
                adjoint_dynamics,
                adjoint_state,
                t.flip(0),
                rtol = adjoint_rtol,
                atol = adjoint_atol,
                method = adjoint_method,
                options = adjoint_options
            )

            # Lấy kết quả cuối cùng tại t_0
            adj_y_t0 = adjoint_outputs[0][-1]
            adj_params_t0 = tuple(out[-1] for out in adjoint_outputs[1:-1])
            adj_t_t0 = adjoint_outputs[-1][-1]

            # Cộng gradient từ các điểm trung gian
            adj_y_t0 = adj_y_t0 + grad_y[0]

            if t_requires_grad:
                time_vjps = adjoint_outputs[-1].flip(0)
            
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
    state_norm = options["norm"]
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
