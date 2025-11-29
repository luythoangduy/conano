# Momentum Prototype-based Local Contrastive Learning

## Tổng quan

Cải tiến Local CL bằng cách sử dụng **momentum encoder** và **prototype bank** (theo phong cách BYOL/MoCo) thay vì random negative sampling.

## Vấn đề với cách cũ

### Global CL
- ✅ Phân biệt tốt giữa các class
- ❌ Làm feature collapse → mất chi tiết để reconstruct

### Local CL (cũ)
- Positive: bắt cặp giữa `q_grid[i,j]` và `k_grid[i, max_sim_idx[j]]`
- Negative: **random** chọn từ class khác
- ❌ Random negatives không ổn định, không đại diện tốt cho class

## Giải pháp mới

### 1. Momentum Encoder (BYOL-style)

**File:** [model/rd.py](model/rd.py:425-441)

```python
# Online network (trainable)
self.proj_layer = MultiProjectionLayer(base=64, dp=dp)

# Momentum network (slow-moving, frozen)
self.proj_layer_momentum = MultiProjectionLayer(base=64, dp=dp)
```

**Update rule:** `θ_momentum = m * θ_momentum + (1-m) * θ_online`

- `m = 0.99` (default)
- Update sau mỗi iteration

### 2. Prototype Bank

**File:** [loss/base_loss.py](loss/base_loss.py:64-83)

```python
self.prototype_bank = {}  # {class_id: feature_map}
```

**Cơ chế:**
- Mỗi class có 1 prototype (được tạo từ momentum encoder)
- **Direct assignment:** `proto = output_of_momentum_encoder`
- Prototypes **ổn định** nhờ momentum encoder (không cần thêm EMA)
- Mỗi iteration update prototype = feature mới nhất từ momentum encoder

### 3. Local CL với Prototypes

**File:** [loss/base_loss.py](loss/base_loss.py:139-183)

#### Positive pairs
Giữ nguyên: `q_grid[i,j]` ↔ `k_grid[i, max_sim_idx[j]]`

#### Negative pairs
**Thay đổi:** Thay vì random sample → dùng **prototypes của class khác**

```python
# Cũ: random negative
neg_k_grid = k_grid[random_different_class]

# Mới: prototype negative
neg_prototype = prototype_bank[different_class_id]
```

**Lợi ích:**
- Negatives ổn định hơn (prototypes thay vì random instances)
- Đại diện tốt hơn cho từng class
- Feature không bị collapse vì prototypes giữ được chi tiết

## Các thay đổi code

### 1. Model ([model/rd.py](model/rd.py))
- ✅ Thêm `proj_layer_momentum`
- ✅ Thêm method `update_momentum_encoder()`
- ✅ Sử dụng momentum encoder cho `k_grid` trong forward

### 2. Loss ([loss/base_loss.py](loss/base_loss.py))
- ✅ Thêm `prototype_bank` và `initialized_classes`
- ✅ Thêm method `update_prototypes()`
- ✅ Sử dụng prototypes làm negatives trong `densecl()`

### 3. Trainer ([trainer/rdlgc_trainer.py](trainer/rdlgc_trainer.py))
- ✅ Gọi `update_momentum_encoder()` sau mỗi iteration

## Hyperparameters

### Momentum cho encoder

#### Option 1: Constant momentum (mặc định cũ)
```python
momentum_schedule = 'constant'
momentum = 0.99  # Fixed value
```

#### Option 2: Progressive momentum (ĐỀ XUẤT MỚI)
```python
momentum_schedule = 'cosine'  # or 'linear', 'step'
momentum_start = 0.9    # Momentum ở đầu training (adaptive)
momentum_end = 0.99     # Momentum ở cuối training (stable)
```

**Lý do sử dụng progressive momentum:**
- 🔄 **Đầu training (m=0.9):** Prototypes adapt nhanh, theo kịp features đang thay đổi
- 🎯 **Giữa training (m=0.95):** Cân bằng giữa stability và adaptability
- 🔒 **Cuối training (m=0.99):** Prototypes rất stable, đại diện tốt cho class

**Các schedule options:**
- `cosine`: Smooth transition, recommended
- `linear`: Linear increase
- `step`: Step-wise increase (0-30%: 0.9, 30-60%: 0.95, 60-100%: 0.99)

### Temperature
```python
temperature = 0.1  # trong DenseLoss.__init__()
```

## Config để sử dụng

Trong file config (ví dụ: `configs/rd/rd_mvtec.py`):

```python
# Model - Progressive Momentum (RECOMMENDED)
model = dict(
    name='rd_lgc',
    kwargs=dict(
        model_t=dict(...),
        model_s=dict(...),
        dp=False,
        momentum_schedule='cosine',   # 'constant', 'linear', 'cosine', 'step'
        momentum_start=0.9,            # Starting momentum
        momentum_end=0.99,             # Ending momentum
    )
)

# Hoặc dùng constant momentum (cách cũ)
model = dict(
    name='rd_lgc',
    kwargs=dict(
        model_t=dict(...),
        model_s=dict(...),
        dp=False,
        momentum_schedule='constant',
        momentum=0.99,
    )
)

# Loss
loss = dict(
    dense=dict(
        name='DenseLoss',
        kwargs=dict(
            lam=1.0,
            temperature=0.1,
            use_prototypes=True,      # Bật prototype mode
        )
    ),
    ...
)
```

## Ablation study gợi ý

1. **Tắt prototypes:** `use_prototypes=False` → so sánh với baseline (random negatives)
2. **Momentum schedules:**
   - Constant: `momentum_schedule='constant', momentum=0.99`
   - Linear: `momentum_schedule='linear', momentum_start=0.9, momentum_end=0.99`
   - Cosine: `momentum_schedule='cosine', momentum_start=0.9, momentum_end=0.99` (RECOMMENDED)
   - Step: `momentum_schedule='step', momentum_start=0.9, momentum_end=0.99`
3. **Momentum ranges:** thử `start=[0.8, 0.9], end=[0.99, 0.999]`
4. **Temperature:** thử `[0.05, 0.1, 0.2]` → control contrastive sharpness

## Kỳ vọng

- 🎯 Feature giữ được chi tiết tốt hơn (cho reconstruction)
- 🎯 Phân biệt class tốt hơn (prototypes stable hơn random)
- 🎯 Training ổn định hơn (momentum update smooth gradients)

## References

- BYOL: Bootstrap Your Own Latent
- MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
- DenseCL: Dense Contrastive Learning for Self-Supervised Visual Pre-Training
