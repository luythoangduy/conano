# Progressive Momentum Scheduler

## Ý tưởng chính

Thay vì dùng **momentum cố định** (0.99), sử dụng **momentum tăng dần** theo thời gian training:

```
Training Progress:  0%  ──────────────────────────────────► 100%
Momentum:          0.9  ──────────────────────────────────► 0.99
                   ↑                                          ↑
              ADAPTIVE                                    STABLE
```

## Tại sao Progressive Momentum hiệu quả?

### Giai đoạn đầu (0-30% training, m=0.9)
**Đặc điểm:**
- Model đang học patterns mới
- Features thay đổi nhanh theo gradient updates
- Prototypes cần UPDATE NHANH để theo kịp

**Với momentum thấp (0.9):**
```python
proto_new = 0.9 * proto_old + 0.1 * feature_new
# 10% contribution từ feature mới → ADAPT nhanh
```

✅ Prototypes linh hoạt, không bị "stuck" ở trạng thái cũ
✅ Theo kịp được features đang thay đổi nhanh

### Giai đoạn giữa (30-60% training, m≈0.95)
**Đặc điểm:**
- Model đã học được phần lớn patterns
- Features bắt đầu converge
- Cần cân bằng giữa stability và adaptability

**Với momentum trung bình (0.95):**
```python
proto_new = 0.95 * proto_old + 0.05 * feature_new
# 5% contribution → Cân bằng
```

✅ Vừa stable vừa adaptive
✅ Smooth transition

### Giai đoạn cuối (60-100% training, m=0.99)
**Đặc điểm:**
- Model converge, features rất ổn định
- Prototypes cần đại diện TỔNG QUÁT cho class
- Tránh overfitting vào noise

**Với momentum cao (0.99):**
```python
proto_new = 0.99 * proto_old + 0.01 * feature_new
# Chỉ 1% contribution → RẤT STABLE
```

✅ Prototypes rất stable, không dao động
✅ Đại diện tốt cho toàn bộ class (average qua nhiều samples)
✅ Robust với noise và outliers

## Các Schedule Options

### 1. Cosine Schedule (RECOMMENDED)
```python
momentum_schedule = 'cosine'
momentum_start = 0.9
momentum_end = 0.99
```

**Đặc điểm:**
- Smooth transition
- Tăng nhanh ở đầu, chậm dần về cuối
- Giống cosine annealing của learning rate

**Công thức:**
```python
progress = iter / max_iter
momentum = end - (end - start) * (1 + cos(π * progress)) / 2
```

**Visualize:**
```
0.99 |                               ╭────────
     |                          ╭────╯
0.95 |                    ╭─────╯
     |              ╭─────╯
0.90 |──────────────╯
     0%           50%              100%
```

### 2. Linear Schedule
```python
momentum_schedule = 'linear'
momentum_start = 0.9
momentum_end = 0.99
```

**Đặc điểm:**
- Đơn giản, dễ hiểu
- Tăng đều đặn

**Công thức:**
```python
progress = iter / max_iter
momentum = start + (end - start) * progress
```

**Visualize:**
```
0.99 |                          ╱
     |                      ╱
0.95 |                  ╱
     |              ╱
0.90 |──────────╱
     0%       50%              100%
```

### 3. Step Schedule
```python
momentum_schedule = 'step'
momentum_start = 0.9
momentum_end = 0.99
```

**Đặc điểm:**
- 3 stages rõ ràng: 0.9 → 0.99 → 0.999
- Easy to analyze và debug
- Phân biệt rõ ràng từng giai đoạn

**Stages:**
- 0-40%: momentum = 0.90 (ADAPTIVE - learn quickly)
- 40-70%: momentum = 0.99 (STABLE - converging)
- 70-100%: momentum = 0.999 (VERY STABLE - finalize prototypes)

**Visualize:**
```
0.999|                         ┌──────────
     |                         │
0.99 |           ┌─────────────┤
     |           │             │
0.90 |───────────┤             │
     0%         40%           70%        100%
```

## So sánh với Constant Momentum

### Constant momentum = 0.99 (baseline)
**Vấn đề:**
- ❌ Đầu training: quá chậm adapt → prototypes lag behind features
- ❌ Prototypes có thể không đại diện tốt trong giai đoạn đầu

### Constant momentum = 0.9 (quá adaptive)
**Vấn đề:**
- ❌ Cuối training: không đủ stable
- ❌ Prototypes dao động theo noise → poor generalization

### Progressive momentum 0.9→0.99 ✅
**Lợi ích:**
- ✅ Best of both worlds
- ✅ Adaptive khi cần, stable khi cần
- ✅ Better balance giữa adaptability và stability

## Config Example

```python
# File: configs/rd/rd_mvtec.py

self.model.kwargs = dict(
    model_t=self.model_t,
    model_s=self.model_s,
    dp=False,

    # Progressive Momentum
    momentum_schedule='cosine',    # Recommended
    momentum_start=0.9,
    momentum_end=0.99,
)
```

## Monitoring

Momentum value sẽ được log trong training:

```
Epoch 10/100:  lr:0.001000  momentum:0.9127  cos:0.123  glb:0.456  dense:0.789
Epoch 50/100:  lr:0.001000  momentum:0.9500  cos:0.098  glb:0.234  dense:0.567
Epoch 100/100: lr:0.001000  momentum:0.9900  cos:0.045  glb:0.123  dense:0.234
```

## Kỳ vọng kết quả

### So với constant momentum:
- **Image AUC:** Tăng (better detection, fewer missed cases)
- **Pixel AUC:** Tăng (better overall accuracy)
- **PRO:** Giữ nguyên hoặc tăng nhẹ (maintain localization quality)

### Lý do:
1. Đầu training: Adaptive momentum → features học nhanh hơn
2. Cuối training: Stable momentum → prototypes generalize tốt hơn
3. Overall: Cân bằng tốt hơn giữa quality và quantity of detections

## Ablation Study

Thử các combinations:

| Schedule | Start | End | Image AUC | Pixel AUC | PRO |
|----------|-------|-----|-----------|-----------|-----|
| constant | - | 0.99 | baseline | baseline | baseline |
| linear | 0.9 | 0.99 | ? | ? | ? |
| cosine | 0.9 | 0.99 | ? | ? | ? |
| step | 0.9 | 0.99 | ? | ? | ? |
| cosine | 0.8 | 0.99 | ? | ? | ? |
| cosine | 0.9 | 0.995 | ? | ? | ? |
