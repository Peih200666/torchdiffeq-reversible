def _interp_fit(y0, y1, y_mid, f0, f1, dt):
    """Hàm này tìm ra hệ số cho đa thức bậc 4.

    Args:
        y0: function value at the start of the interval(khoảng).
        y1: function value at the end of the interval.
        y_mid: function value at the mid-point of the interval.
        f0: derivative value at the start of the interval.
        f1: derivative value at the end of the interval.
        dt: width of the interval.

    Returns:
        List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
        `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
        between 0 (start of interval) and 1 (end of interval).
    """
    a = 2 * dt * (f1 - f0) - 8 * (y1 + y0) + 16 * y_mid
    b = dt * (5 * f0 - 3 * f1) + 18 * y0 + 14 * y1 - 32 * y_mid
    c = dt * (f1 - 4 * f0) - 11 * y0 - 5 * y1 + 16 * y_mid
    d = dt * f0
    e = y0
    return [e, d, c, b, a] 
    # đảm bảo đường nội suy trơn và chính xác ở 3 điểm (đầu, giữa, cuối).


def _interp_evaluate(coefficients, t0, t1, t):
    """ Tính giá trị của đường cong nội suy tại một thời điểm bất kỳ

    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar float64 Tensor giving the start of the interval.
        t1: scalar float64 Tensor giving the end of the interval.
        t: Thời điểm mà ta muốn nội suy (phải nằm giữa t0 và t1)

    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """

    assert (t0 <= t) & (t <= t1), 'invalid interpolation, fails `t0 <= t <= t1`: {}, {}, {}'.format(t0, t, t1)
    # kiểm tra điều kiện: đúng → chương trình chạy tiếp, sai → báo lỗi ngay lập tức
    # .format() chỉ là cách điền giá trị biến vào chuỗi
    x = (t - t0) / (t1 - t0) # chuẩn hóa (normalize) giá trị thời gian t sang thang chuẩn [0, 1] -> toi uu hon, de kiem soat sai so.
    x = x.to(coefficients[0].dtype) # change data type of x -> same type with coeffictiens

    total = coefficients[0] + x * coefficients[1] # total = e + x * d
    x_power = x
    for coefficient in coefficients[2:]: # 2: bỏ qua 2 phần đầu
        x_power = x_power * x
        total = total + x_power * coefficient
    # total = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e
    return total
