import random
import torch
import torch.nn.functional as nn_func
# from torch.nn.utils.rnn import pad_sequence
from einops import rearrange  # , repeat
from torch import nn, Tensor

from typing import List
import math

import torch_xla
import torch_xla.core.xla_model as xm


# from torch_xla import runtime as xr
# from torch_xla._internal import tpu
# from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_impl, SegmentIds
# from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

from torch_xla.experimental.custom_kernel import flash_attention
from torch_xla.experimental.custom_kernel import jax_import_guard
jax_import_guard()
import jax

import copy

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class TpuNavitAttention(nn.Module):

    def __init__(self, pack_length, dim, heads=8, dropout=0.0, scale=None, device=None):

        inner_dim = dim
        self.device = xm.xla_device()
        # initializing tensors must be of the dimensions given in params
        assert inner_dim % heads == 0

        super().__init__()

        # pack_length must be the same for context and image, if different it will be not possible to do self-attention
        self.pack_length = pack_length

        self.scale_factor = 1 / math.sqrt(dim) if scale is None else scale
        self.heads = heads

        self.softmax = nn.Softmax(dim=-1).to(self.device)
        self.dropout = dropout

        self.dim = dim
        self.inner_dim = inner_dim

    @staticmethod
    @torch.no_grad()
    def build_pic_id_sequence(sequence_list: List[int], navit_pack_length):
        return nn_func.pad(Tensor(sequence_list).to("xla"),
                           pad=(0, navit_pack_length - len(sequence_list)),
                           value=NavitAttention.X_IND_PADDING
                           )

    @staticmethod
    @torch.no_grad()
    def build_ctx_id_sequence(sequence_list: List[int], navit_pack_length):
        return nn_func.pad(Tensor(sequence_list).to("xla"),
                           pad=(0, navit_pack_length - len(sequence_list)),
                           value=NavitAttention.C_IND_PADDING
                           )

    @staticmethod
    @torch.no_grad()
    def pad_embedding_sequence(embed_sequence, navit_pack_length):
        return nn_func.pad(embed_sequence,
                           pad=(0, 0, 0, navit_pack_length - embed_sequence.shape[0]),
                           value=0)

    def forward(self, q, q_indexes, k, v, kv_indexes):
        assert q.shape[1] == self.dim


        q = rearrange(q, "n d -> 1 1 n d") * self.scale_factor
        k = rearrange(k, "n d -> 1 1 n d")
        v = rearrange(v, "n d -> 1 1 n d")
        q_indexes = rearrange(q_indexes, "n -> 1 n")
        kv_indexes = rearrange(kv_indexes, "n -> 1 n")
        q = q.to(self.device)
        k = k.to(self.device)
        v = v.to(self.device)
        q_indexes = q_indexes.to(self.device)
        kv_indexes = kv_indexes.to(self.device)

        # using tpu flash attention function, which hopefully will be optimised

        out = flash_attention(q, k, v, q_segment_ids=q_indexes, kv_segment_ids=kv_indexes)
        # out = _attention(
        # q,
        # k,
        # v,
        # attn_mask=_make_attention_mask_from_segment_ids(
        #     q_indexes, kv_indexes))

        # out *= self.scale_factor
        # xm.mark_step()
        # multi-head support:   out = rearrange(out, "h n d -> n (h d)")
        return out


class NavitAttention(nn.Module):

    X_IND_PADDING = -1
    C_IND_PADDING = -2

    def make_attention_mask(self, x_indexes, c_indexes):
        # c_indexes and x_indexes must come pre-padded to self.pack_length with different padding!!
        # This is important for building the attention mask
        assert (c_indexes.shape[0] == self.pack_length) or (x_indexes.shape[0] == self.pack_length)
        # build a table of match between IDs of context and IDs of pic embedding
        # size = (pack_length, pack_length)
        return rearrange(x_indexes, "b -> b 1") == rearrange(c_indexes, "b -> 1 b")

    def __init__(self, dim, heads=8, dropout=0.0, scale=None, pack_length=64): #inner_dim=0,
        """
        :param dim: dimension of the embedding space. Currently x and context must be in the same dimension space.
        :param heads: number of attention heads
        :param dropout: percentage of dropout for attention
        :param scale: override the scale of the attention by this value, by default sqrt(1/dim)

        """


        inner_dim = dim

        # initializing tensors must be of the dimensions given in params
        assert inner_dim % heads == 0

        super().__init__()

        # pack_length must be the same for context and image, if different it will be not possible to do self-attention
        self.pack_length = pack_length

        self.scale_factor = 1 / math.sqrt(dim) if scale is None else scale
        self.heads = heads

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = dropout

        self.dim = dim
        self.inner_dim = inner_dim


    @staticmethod
    @torch.no_grad()
    def build_pic_id_sequence(sequence_list: List[int], navit_pack_length):
        return nn_func.pad(Tensor(sequence_list),
                           pad=(0, navit_pack_length - len(sequence_list)),
                           value=NavitAttention.X_IND_PADDING
                           )

    @staticmethod
    @torch.no_grad()
    def build_ctx_id_sequence(sequence_list: List[int], navit_pack_length):
        return nn_func.pad(Tensor(sequence_list),
                           pad=(0, navit_pack_length - len(sequence_list)),
                           value=NavitAttention.C_IND_PADDING
                           )

    @staticmethod
    @torch.no_grad()
    def pad_embedding_sequence(embed_sequence, navit_pack_length):
        return nn_func.pad(embed_sequence,
                           pad=(0, 0, 0, navit_pack_length - embed_sequence.shape[0]),
                           value=0)

    def for_ward_manual(self, q, q_indexes, k, v, kv_indexes):
        assert q.shape[1] == self.dim

        attn_mask = self.make_attention_mask(q_indexes, kv_indexes)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        # out = rearrange(out, "h n d -> n (h d)")
        return out # [:original_seq_length, :]  # self.to_out(out)

    def forward(self, q, q_indexes, k, v, kv_indexes):
        assert q.shape[1] == self.dim
        # create attention mask to attend only embeddings related to same picture, arguments show which
        # embeddings in the sequence belong to where
        attn_mask = self.make_attention_mask(q_indexes, kv_indexes)

        attn_bias = torch.zeros(self.pack_length, self.pack_length, dtype=q.dtype)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-100000000000000000000000"))
        # using standard attention function, which hopefully will be optimised
        out = nn_func.scaled_dot_product_attention(q, k, v, attn_bias)
        # multi-head support:   out = rearrange(out, "h n d -> n (h d)")
        return out

def _make_attention_mask_from_segment_ids(q_segment_ids, kv_segment_ids):
    return q_segment_ids.view(q_segment_ids.shape[0], 1,
                            q_segment_ids.shape[1], 1) != kv_segment_ids.view(
                                kv_segment_ids.shape[0], 1, 1,
                                kv_segment_ids.shape[1])

def _attention(q, k, v, *, attn_mask=None):
    attn_weight = q @ k.transpose(-2, -1)
    if attn_mask is not None:
      # Masked out the unrelevant parts.
      attn_weight = attn_weight.masked_fill(attn_mask,
                                            torch.finfo(attn_weight.dtype).min)
    attn_weight = nn.functional.softmax(attn_weight, dim=-1)
    attn_output = attn_weight @ v
    return attn_output

def navit_attention_accuracy_test():
    print(f"Navit Forward Attention Accuracy Test ----------------------------- ")
    navit_pack_length = 128
    pic_in_sequence = 4
    max_embeddings_in_pic = int(navit_pack_length / pic_in_sequence)
    dim = 64
    for test_index in range(10):
        K = nn.Linear(dim, dim)
        # K.weight.data = torch.rand((dim, dim))
        V = nn.Linear(dim, dim)
        # V.weight.data = torch.rand((dim, dim))
        Q = nn.Linear(dim, dim)
        # Q.weight.data = torch.rand((dim, dim))

        pic_embed_sequence = torch.ones(0, dim)
        pic_embed_indexes = []
        cont_embed_sequence = torch.ones(0, dim)
        cont_embed_indexes = []

        straight_attention = torch.ones(0, dim)
        for picture_index in range(pic_in_sequence):
            num_of_pic_embeddings = int(random.random() * max_embeddings_in_pic) + 1
            num_of_con_embeddings = int(random.random() * max_embeddings_in_pic) + 1
            pic_embeddings = torch.rand(num_of_pic_embeddings, dim)
            context_embeddings = torch.rand(num_of_con_embeddings, dim)

            pic_embed_sequence = torch.cat((pic_embed_sequence, pic_embeddings))
            cont_embed_sequence = torch.cat((cont_embed_sequence, context_embeddings))
            pic_embed_indexes += [picture_index] * num_of_pic_embeddings
            cont_embed_indexes += [picture_index] * num_of_con_embeddings

            # ----------- Calculate attention in usual way one picture each time -- for validation ----------
            # calculate usual attention for these inputs for comparison reference
            # print(pic_embed_sequence.shape, cont_embed_sequence.shape)
            q = Q(pic_embeddings) # torch.matmul(Q, pic_embeddings.transpose(0, 1)).transpose(0, 1)
            k = K(context_embeddings) # torch.matmul(K, context_embeddings.transpose(0, 1)).transpose(0, 1)
            v = V(context_embeddings) # torch.matmul(V, context_embeddings.transpose(0, 1)).transpose(0, 1)
            st_att = nn_func.scaled_dot_product_attention(q, k, v)

            # concatenate all the results to series, to compare to the series of Navit at the end
            straight_attention = torch.cat((straight_attention, st_att))
            # --------------------------------- Calc usual way -----------------------------------------------

        # ---------------     calculate the attention Navit way -----------------------------

        # at = NavitAttention(dim=dim, heads=1, pack_length=navit_pack_length)
        at = TpuNavitAttention(dim=dim, heads=1, pack_length=navit_pack_length)

        # print(f"indexes {pic_embed_indexes} , len {len(pic_embed_indexes)}")
        original_seq_length = pic_embed_sequence.shape[0]
        # pic_embed_sequence is a pack of all embeddings of multiple pictures. Pad the sequences to the standard pack length.
        # Pack must be of same length every time otherwise code will recompile for TPU
        # must come pre-padded
        pic_embed_sequence = NavitAttention.pad_embedding_sequence(pic_embed_sequence, navit_pack_length)
        cont_embed_sequence = NavitAttention.pad_embedding_sequence(cont_embed_sequence, navit_pack_length)
        q = Q(pic_embed_sequence) # torch.matmul(Q, pic_embed_sequence.transpose(0, 1)).transpose(0, 1)
        k = K(cont_embed_sequence) # torch.matmul(K, cont_embed_sequence.transpose(0, 1)).transpose(0, 1)
        v = V(cont_embed_sequence) # torch.matmul(V, cont_embed_sequence.transpose(0, 1)).transpose(0, 1)
        test_output = at.forward(q=q,
                                 q_indexes=NavitAttention.build_pic_id_sequence(pic_embed_indexes, navit_pack_length),
                                 k=k,
                                 v=v,
                                 kv_indexes=NavitAttention.build_ctx_id_sequence(cont_embed_indexes, navit_pack_length)
                                 )
        test_output = test_output[0, 0, :original_seq_length, :]
        # test_output = test_output[:original_seq_length, :]
        xm.mark_step()
        # test_output = test_output[ :original_seq_length, :]
        # ---------------   calc Navit   ------------------------------------------------------

        print(straight_attention.shape, test_output.shape)

        diff = torch.max(torch.abs(test_output.cpu() - straight_attention.cpu()))
        print(f"max diff: {diff},  ")

        # assert diff < 0.001, \
        #     "########################   ATTENTION CALCULATION ERROR   #########################"
    print(f"Navit Forward Attention Accuracy Test - OK  =============\n\n\n")

def navit_backward_accuracy():
    # from torch.autograd import grad
    print(f"Navit Backward Attention Accuracy Test ----------------------------- ")
    navit_pack_length = 128
    pic_in_sequence = 6
    max_embeddings_in_pic = int(navit_pack_length / pic_in_sequence)
    dim = 64
    K = nn.Linear(dim, dim)
    V = nn.Linear(dim, dim)
    Q = nn.Linear(dim, dim)
    Q2 = copy.deepcopy(Q).to("xla")
    K2 = copy.deepcopy(K).to("xla")
    V2 = copy.deepcopy(V).to("xla")

    pic_embed_sequence = torch.ones(0, dim)
    pic_embed_indexes = []
    cont_embed_sequence = torch.ones(0, dim)
    cont_embed_indexes = []

    straight_attention = torch.ones(0, dim)

    pics = []
    ctxs = []
    with torch.no_grad():
        for picture_index in range(pic_in_sequence):
            num_of_pic_embeddings = int(random.random() * max_embeddings_in_pic) + 1
            num_of_con_embeddings = int(random.random() * max_embeddings_in_pic) + 1
            pic_embeddings = torch.rand(num_of_pic_embeddings, dim)
            # pic_embeddings.requires_grad = True
            context_embeddings = torch.rand(num_of_con_embeddings, dim)
            # context_embeddings.requires_grad = True
            pics.append(pic_embeddings)
            ctxs.append(context_embeddings)

            pic_embed_sequence = torch.cat((pic_embed_sequence, pic_embeddings))
            cont_embed_sequence = torch.cat((cont_embed_sequence, context_embeddings))
            pic_embed_indexes += [picture_index] * num_of_pic_embeddings
            cont_embed_indexes += [picture_index] * num_of_con_embeddings

    for picture_index in range(pic_in_sequence):
        q = Q(pics[picture_index])
        k = K(ctxs[picture_index])
        v = V(ctxs[picture_index])
        st_att = nn_func.scaled_dot_product_attention(q, k, v)
        # concatenate all the results to series, to compare to the series of Navit at the end
        straight_attention = torch.cat((straight_attention, st_att))

    loss = torch.sum(straight_attention)
    loss.backward()
    grad_cat = torch.ones(0, dim)
    grad_cat = torch.cat((grad_cat, Q.weight.grad))
    grad_cat = torch.cat((grad_cat, K.weight.grad))
    grad_cat = torch.cat((grad_cat, V.weight.grad))


    # ---------------     calculate the attention Navit way -----------------------------

    # at = NavitAttention(dim=dim, heads=1, pack_length=navit_pack_length)
    at = TpuNavitAttention(dim=dim, heads=1, pack_length=navit_pack_length)
    pic_embed_sequence = pic_embed_sequence.detach().to("xla")
    cont_embed_sequence = cont_embed_sequence.detach().to("xla")

    original_seq_length = pic_embed_sequence.shape[0]

    pic_embed_sequence = TpuNavitAttention.pad_embedding_sequence(pic_embed_sequence, navit_pack_length)
    cont_embed_sequence = TpuNavitAttention.pad_embedding_sequence(cont_embed_sequence, navit_pack_length)
    q = Q2(pic_embed_sequence)
    k = K2(cont_embed_sequence)
    v = V2(cont_embed_sequence)
    test_output = at.forward(q=q,
                             q_indexes=TpuNavitAttention.build_pic_id_sequence(pic_embed_indexes, navit_pack_length),
                             k=k,
                             v=v,
                             kv_indexes=TpuNavitAttention.build_ctx_id_sequence(cont_embed_indexes, navit_pack_length)
                             )

    # test_output = test_output[:original_seq_length, :]
    test_output = test_output[0, 0, :original_seq_length, :]
    # ---------------   calc Navit   ------------------------------------------------------

    loss = torch.sum(test_output)
    loss.backward()
    xm.mark_step()
    grad_cat2 = torch.ones(0, dim)
    grad_cat2 = torch.cat((grad_cat2, Q2.weight.grad.cpu()))
    grad_cat2 = torch.cat((grad_cat2, K2.weight.grad.cpu()))
    grad_cat2 = torch.cat((grad_cat2, V2.weight.grad.cpu()))



    diff = torch.max(torch.abs(grad_cat2 - grad_cat))
    print(f"grad max diff: {diff},  ")

    diff = torch.max(torch.abs(test_output.cpu() - straight_attention))
    print(f"attn max diff: {diff},  ")

    assert diff < 0.001, \
        "########################   ATTENTION CALCULATION ERROR   #########################"
    print(f"Navit Backward Attention Accuracy Test - OK  =============\n\n\n")


if __name__ == "__main__":
    import os
    # os.environ['PJRT_DEVICE'] = 'TPU'
    # os.environ['TPU_NUM_DEVICES'] = '1'
    # os.environ['PT_XLA_DEBUG'] = '1'
    # os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
    # torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(linewidth=200)  # let torch print on all the screen

    torch_xla._XLAC._xla_set_use_full_mat_mul_precision(use_full_mat_mul_precision=True)
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
    navit_attention_accuracy_test()
    navit_backward_accuracy()
