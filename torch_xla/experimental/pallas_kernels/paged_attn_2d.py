# DO NOT REVIEW. DO NOT CHANGE.
# This is the reference extended paged attention.
from typing import List

import torch


def ref_paged_attn(
    q: torch.Tensor,                # [batch_size, query_len, num_query_heads, head_size]
    k_pages: torch.Tensor,          # [num_kv_heads, total_num_pages, page_size, head_size]
    v_pages: torch.Tensor,          # [num_kv_heads, total_num_pages, page_size, head_size]
    lengths: torch.Tensor,          # [batch_size]
    page_indices: torch.Tensor,     # [batch_size, pages_per_sequence]
) -> torch.Tensor:                  # [batch_size, query_len, num_query_heads, head_size]
    batch_size, query_len, num_query_heads, head_size = q.shape
    num_kv_heads, total_num_pages, page_size, _ = k_pages.shape
    num_query_per_kv = num_query_heads // num_kv_heads

    lengths = lengths.cpu()
    page_indices = page_indices.cpu()

    outputs: List[torch.Tensor] = []
    for i in range(batch_size):
        kv_len = lengths[i]
        num_pages = (kv_len + page_size - 1) // page_size
        indices = page_indices[i, :num_pages]

        k = k_pages[:, indices]     # [num_kv_heads, num_pages, page_size, head_size]
        k = k.permute(1, 2, 0, 3)   # [num_pages, page_size, num_kv_heads, head_size]
        k = k.reshape(num_pages * page_size, num_kv_heads, head_size)
        k = k[:kv_len]              # [kv_len, num_kv_heads, head_size]

        v = v_pages[:, indices]     # [num_kv_heads, num_pages, page_size, head_size]
        v = v.permute(1, 2, 0, 3)   # [num_pages, page_size, num_kv_heads, head_size]
        v = v.reshape(num_pages * page_size, num_kv_heads, head_size)
        v = v[:kv_len]              # [kv_len, num_kv_heads, head_size]

        if num_query_per_kv != 1:
            # GQA/MQA
            k = torch.repeat_interleave(k, num_query_per_kv, dim=1)     # [kv_len, num_query_heads, head_size]
            v = torch.repeat_interleave(v, num_query_per_kv, dim=1)     # [kv_len, num_query_heads, head_size]

        # NOTE: To balance efficiency and performance, we use the original dtype (e.g., bfloat16 or float16)
        # for matrix multiplications (i.e., q @ k and attn @ v) while using float32 for softmax.
        # However, the kernel doesn't have to strictly follow the dtypes here.
        # For example, it can use bfloat16 instead of float32 or vice versa for performance or simplicity.
        # Note, "qhd,khd->hqk" corresponds to
        # [query_len, num_query_heads, head_size],[kv_len, num_query_heads, head_size] -> 
        # [num_query_heads,query_len,kv_len]
        # So below is: q[i].permute(1,0,2)@k.permute(1,2,0) aka:
        # [num_query_heads, query_len, head_size]@[num_query_heads, head_size, kv_len]->[num_query_heads,query_len,kv_len]
        # N.B in Karpathy's impl, it's wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        attn = torch.einsum("qhd,khd->hqk", q[i], k)       # [num_query_heads, query_len, kv_len]
        attn = attn.float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)  # [num_query_heads, query_len, kv_len]
        # N.B in Karpathy's impl, it's out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        # [num_query_heads, query_len, kv_len]X[kv_len, num_query_heads, head_size]->
        # [query_len,num_query_heads,head_size]
        # Does it mean: temp=(attn @ v.permute(1,0,2)) [num_query_heads, query_len, head_size], then
        # temp.permute(1,0,2)
        out = torch.einsum("hqk,khd->qhd", attn, v)     # [query_len, num_query_heads, head_size]
        outputs.append(out)

    output = torch.stack(outputs, dim=0)    # [batch_size, query_len, num_query_heads, head_size]
    return output


@torch.no_grad()
def test_paged_attn(
    batch_size: int = 3,
    query_len: int = 128,
    num_query_heads: int = 64,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    max_kv_len: int = 1024,
    page_size: int = 16,
    total_num_pages: int = 1000,
    device: str = "cpu",
) -> None:
    assert num_query_heads % num_kv_heads == 0
    assert query_len <= max_kv_len
    assert max_kv_len <= total_num_pages * page_size

    with torch.device(device):
        query = torch.randn(batch_size, query_len, num_query_heads, head_size, dtype=dtype)
        k_pages = torch.randn(num_kv_heads, total_num_pages, page_size, head_size, dtype=dtype)
        v_pages = torch.rand_like(k_pages)
        lengths = torch.randint(query_len, max_kv_len + 1, (batch_size,))
        page_indices = torch.randint(0, total_num_pages, (batch_size, total_num_pages))

        # NOTE: The reference implementation does not include the scaling factor.
        scale = 1.0 / (head_size ** 0.5)
        scaled_query = query * scale
        ref_output = ref_paged_attn(scaled_query, k_pages, v_pages, lengths, page_indices)
        print(ref_output)


if __name__ == "__main__":
    test_paged_attn()
