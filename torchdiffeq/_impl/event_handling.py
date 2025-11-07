import math
import torch


def find_event(interp_fn, sign0, t0, t1, event_fn, tol):
    # intern_fn: Hàm nội suy (ước lượng giá trị ở giữa các điểm đã biết)
    # find_event: tìm thời điểm xảy ra sự kiện (khi event_fn = 0)
    with torch.no_grad(): # Tắt chế độ tính đạo hàm
        # Cần lặp bao nhiêu lần chia đôi de thu hẹp thời gian <= sai so tol
        # L₀ = (t1 - t0) (độ dài khoảng ban dau)
        # Sau n lần chia đôi: Ln​ = L₀ / (2 ^ n) <= tol
        # -> (2 ^ n) = L₀ / tol -> n >= log2​(L₀ / tol)
        nitrs = torch.ceil(torch.log((t1 - t0)  / tol) / math.log(2.0))
        # torch.log((t1 - t0) / tol) →  ln(L₀ / tol)
        # math.log(2.0) -> đổi log tự nhiên thành log cơ số 2 (default: log_e)
        # torch.celi: Làm tròn lên số nguyên gần nhất

        for _ in range(nitrs.long()):
            t_mid = (t1 + t0) / 2.0
            y_mid = interp_fn(t_mid)
            sign_mid = torch.sign(event_fn(t_mid, y_mid)) # Tính dấu tại điểm giữa
            same_as_sign0 = (sign0 == sign_mid) 
            t0 = torch.where(same_as_sign0, t_mid, t0) #torch.where(condition, A, B) condition = True -> A, False -> B
            t1 = torch.where(same_as_sign0, t1, t_mid)
        event_t = (t0 + t1) / 2.0

    return event_t, interp_fn(event_t)


def combine_event_functions(event_fn, t0, y0):
    """
    We ensure all event functions are initially positive,
    so then we can combine them by taking a min.
    """
    with torch.no_grad():
        initial_signs = torch.sign(event_fn(t0, y0)) # dấu ban đầu của từng event

    def combined_event_fn(t, y):
        c = event_fn(t, y)
        return torch.min(c * initial_signs) # Giúp tất cả event bắt đầu ở vùng dương

    return combined_event_fn


def interp_fn(t):
    return torch.tensor([10.0 - 3 * t])   # trả về trạng thái tại thời điểm t

# Hàm sự kiện: dừng khi y = 0
def event_fn(t, y):
    return y  # event xảy ra khi y = 0

t0 = torch.tensor([-2.0])
t1 = torch.tensor([10.0])
tol = torch.tensor([1e-10])
y0 = interp_fn(t0)
sign0 = torch.sign(event_fn(t0, y0))

event_t, y_event = find_event(interp_fn, sign0, t0, t1, event_fn, tol)

print(f"Thời điểm sự kiện xảy ra: {event_t.item():.4f}")
print(f"Trạng thái y tại thời điểm đó: {y_event.item():.4f}")

def multi_event_fn(t, y):
    return torch.stack([1.0 - t, 3.0 - t])

combined_event = combine_event_functions(multi_event_fn, t0, y0)

for test_t in [0.5, 1.0, 1.5, 2.0]:
    val = combined_event(torch.tensor([test_t]), None)
    print(f"t={test_t:.1f}, combined_event={val.item():.4f}")