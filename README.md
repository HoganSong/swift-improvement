# 사전 과제 메인 보고서
<br>

## 1. ​대규모 토큰 사용 과정에서의 한계점

**Mask Caching이 없다는 점**
> BasicLayer의 forward가 호출될 때마다 compute_mask 함수가 돌아가며 mask를 다시 계산하면서 성능 저하가 발생한다. 특히 대규모 토큰이 입력될 시 mask의 크기도 그에 따라 증가하기 때문에 성능 저하가 증폭된다.
<br>

**핵심 프롬프트**
+ What are the large input limitations during inference, specific to this SwinTransformer4D model?
+ I'd like to focus on 2. Lack of Mask Caching, and 5. Memory Copying in 4D Partitioning, since 1., 3., and 4. emerges from deliberate design choices, and 6. seems like an edge case. Is this true?
<br>

## 2. 문제 해결 전략

1. **Global Mask Caching을 구현**
> forward 내부에 있던 compute_mask의 기능을 토대로 외부의 global mask cache를 활용할 수 있도록 했다. 이를 통해 forward가 호출될 때마다 mask를 새롭게 만드는 것이 아니라 평소에는 global mask cache로부터 mask를 갖고 오되, 필요 시에만 새로 계산하도록 바뀌었다.

2. **Boolean Mask 활용**
> Mask 값들을 기본적으로 float32인 torch.zeros를 통해서가 아니라 torch.bool 값으로 저장하도록 하여, 메모리 활용을 경감시켰다.
<br>

**핵심 코드**
```
_ATTN_MASK_CACHE = {}

def get_attn_mask(dims, window_size, shift_size, device):
    """
    Retrieves or computes a boolean attention mask.
    Returns True for positions that should be masked (-inf).
    """
    key = (tuple(dims), tuple(window_size), tuple(shift_size), str(device))
    if key not in _ATTN_MASK_CACHE:
        if len(_ATTN_MASK_CACHE) > 50:
            _ATTN_MASK_CACHE.clear()
        
        # Compute original mask logic
        d_dim, h_dim, w_dim, t_dim = dims
        img_mask = torch.zeros((1, d_dim, h_dim, w_dim, t_dim, 1), device=device)
        cnt = 0
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    for t in slice(-window_size[3]), slice(-window_size[3], -shift_size[3]), slice(-shift_size[3], None):
                        img_mask[:, d, h, w, t, :] = cnt
                        cnt += 1

        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.squeeze(-1)
        # Create boolean mask: True where values are different (indicating a cross-boundary attention)
        attn_mask = (mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2))
        _ATTN_MASK_CACHE[key] = attn_mask
        
    return _ATTN_MASK_CACHE[key]
```
<br>

## 3. 실험 및 결과

**실험 내용**
> dummy input을 입력하여 수정 전후의 모델을 평균 구동 시간 및 gpu 메모리 활용 정도의 관점에서 비교 분석했다. 구동 시간은 warm-up pass 제외 20번 구동되는 평균 시간을 지표로 삼았으며, gpu 메모리 활용 정도는 gpu에 할당된 최대 메모리 값을 지표로 삼았다.

**실험 결과**
> 수정 전에 비해 구동 속도가 13.0% 정도 증가했으며, 최대 메모리 사용량은 8.8% 정도 감소했다.

![terminal result](/result.png)
