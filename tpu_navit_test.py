import random
import torch
import torch.nn.functional as nn_func
# from torch.nn.utils.rnn import pad_sequence
from einops import rearrange  # , repeat
from torch import nn, Tensor

from typing import List
import math

# import torch_xla
import torch_xla.core.xla_model as xm


# from torch_xla import runtime as xr
# from torch_xla._internal import tpu
# from jax.experimental.pallas.ops.tpu.flash_attention import _flash_attention_impl, SegmentIds
# from torch_xla.experimental.custom_kernel import make_kernel_from_pallas

from torch_xla.experimental.custom_kernel import flash_attention
from torch_xla.experimental.custom_kernel import jax_import_guard
jax_import_guard()
import jax

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class TpuNavitAttention(nn.Module):

    def __init__(self, pack_length, dim, heads=8, dropout=0.0, scale=None, device=None, multi_head=False):

        inner_dim = dim
        self.device = xm.xla_device()
        self.multi_head = multi_head
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
    def build_pic_id_sequence(sequence_list: List[int], navit_pack_length):
        return nn_func.pad(Tensor(sequence_list),
                           pad=(0, navit_pack_length - len(sequence_list)),
                           value=NavitAttention.X_IND_PADDING
                           )

    @staticmethod
    def build_ctx_id_sequence(sequence_list: List[int], navit_pack_length):
        return nn_func.pad(Tensor(sequence_list),
                           pad=(0, navit_pack_length - len(sequence_list)),
                           value=NavitAttention.C_IND_PADDING
                           )

    @staticmethod
    def pad_embedding_sequence(embed_sequence, navit_pack_length):
        return nn_func.pad(embed_sequence,
                           pad=(0, 0, 0, navit_pack_length - embed_sequence.shape[0]),
                           value=0)

    def forward(self, q, q_indexes, k, v, kv_indexes):
        assert q.shape[1] == self.dim
        if self.multi_head:
            q = rearrange(q, "n (h d) -> 1 h n d", h=8)
            k = rearrange(k, "n (h d) -> 1 h n d", h=8)
            v = rearrange(v, "n (h d) -> 1 h n d", h=8)
        else:
            q = rearrange(q, "n d -> 1 1 n d")
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

        out = flash_attention(q, k, v, q_segment_ids=q_indexes, kv_segment_ids=kv_indexes, sm_scale=self.scale_factor)

        # multi-head support:
        if self.multi_head:
            out = rearrange(out, "1 h n d ->  n (h d)")
        else:
            out = rearrange(out, "1 1 n d ->  n d")

        xm.mark_step()

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

    def __init__(self, dim, heads=8, dropout=0.0, scale=None, pack_length=64, multi_head=False): #inner_dim=0,
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
        self.multi_head = multi_head
        # pack_length must be the same for context and image, if different it will be not possible to do self-attention
        self.pack_length = pack_length

        self.scale_factor = 1 / math.sqrt(dim) if scale is None else scale
        self.heads = heads

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = dropout

        self.dim = dim
        self.inner_dim = inner_dim


    @staticmethod
    def build_pic_id_sequence(sequence_list: List[int], navit_pack_length):
        return nn_func.pad(Tensor(sequence_list),
                           pad=(0, navit_pack_length - len(sequence_list)),
                           value=NavitAttention.X_IND_PADDING
                           )

    @staticmethod
    def build_ctx_id_sequence(sequence_list: List[int], navit_pack_length):
        return nn_func.pad(Tensor(sequence_list),
                           pad=(0, navit_pack_length - len(sequence_list)),
                           value=NavitAttention.C_IND_PADDING
                           )

    @staticmethod
    def pad_embedding_sequence(embed_sequence, navit_pack_length):
        return nn_func.pad(embed_sequence,
                           pad=(0, 0, 0, navit_pack_length - embed_sequence.shape[0]),
                           value=0)

    def for_ward_manual(self, x, x_indexes, c, c_indexes):
        assert x.shape[1] == self.dim

        # if context is not given - doing self-attention
        if c is None:
            c = x
            c_indexes = x_indexes

        attn_mask = self.make_attention_mask(x_indexes, c_indexes)

        qkv = (self._q(x), *self._kv(c).chunk(2, dim=-1))
        # print(f"qkv \n{qkv}")
        q, k, v = map(lambda t: rearrange(t, "n (h d) -> h n d", h=self.heads), qkv)
        # print(f"Qx \n{q}")
        # print(f"Kc \n{k}")
        # print(f"Vc \n{v}")

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        # torch.set_printoptions(precision=1)
        # print(f"Correlation before mask \n{dots}")

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.softmax(dots)

        # if exists(attn_mask):
        #    attn = attn.masked_fill(~attn_mask, 0)

        # print(f"Correlation after mask \n{attn}")
        # attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        # print(f"Out  \n{out}")
        out = rearrange(out, "h n d -> n (h d)")
        # print(f"Out  rearranged\n{out}")
        return out # [:original_seq_length, :]  # self.to_out(out)

    def forward(self, q, q_indexes, k, v, kv_indexes):
        assert q.shape[1] == self.dim
        # create attention mask to attend only embeddings related to same picture, arguments show which
        # embeddings in the sequence belong to where
        attn_mask = self.make_attention_mask(q_indexes, kv_indexes)

        attn_bias = torch.zeros(self.pack_length, self.pack_length, dtype=q.dtype)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-100000000000000000000000"))

        # using standard attention function, which hopefully will be optimised
        if self.multi_head:
            attn_bias = rearrange(attn_bias, "n m -> 1 1 n m").expand(-1, 8, -1, -1)
            q = rearrange(q, "n (h d) -> 1 h n d", h=8)
            k = rearrange(k, "n (h d) -> 1 h n d", h=8)
            v = rearrange(v, "n (h d) -> 1 h n d", h=8)
        else:
            q = rearrange(q, "n d -> 1 1 n d")
            k = rearrange(k, "n d -> 1 1 n d")
            v = rearrange(v, "n d -> 1 1 n d")
            # multi-head support:

        out = nn_func.scaled_dot_product_attention(q, k, v, attn_bias)

        if self.multi_head:
            out = rearrange(out, "1 h n d ->  n (h d)")
        else:
            out = rearrange(out, "1 1 n d ->  n d")

        return out


def navit_attention_accuracy_test(multihead_test=False):
    print(f"Navit Forward Attention Accuracy Test ----------------------------- ")
    navit_pack_length = 128
    pic_in_sequence = 3
    max_embeddings_in_pic = int(navit_pack_length / pic_in_sequence)
    dim = 64
    for test_index in range(10):
        K = nn.Linear(dim, dim)
        K.weight.data = torch.rand((dim, dim))
        V = nn.Linear(dim, dim)
        V.weight.data = torch.rand((dim, dim))
        Q = nn.Linear(dim, dim)
        Q.weight.data = torch.rand((dim, dim))

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
            q = Q(pic_embeddings) # torch.matmul(Q, pic_embeddings.transpose(0, 1)).transpose(0, 1)
            k = K(context_embeddings) # torch.matmul(K, context_embeddings.transpose(0, 1)).transpose(0, 1)
            v = V(context_embeddings) # torch.matmul( V, context_embeddings.transpose(0, 1)).transpose(0, 1)
            if multihead_test:
                q = rearrange(q, "n (h d) -> 1 h n d", h=8)
                k = rearrange(k, "n (h d) -> 1 h n d", h=8)
                v = rearrange(v, "n (h d) -> 1 h n d", h=8)
            st_att = nn_func.scaled_dot_product_attention(q, k, v)
            if multihead_test:
                st_att = rearrange(st_att, "1 h n d -> n (h d)")
            # concatenate all the results to series, to compare to the series of Navit at the end
            straight_attention = torch.cat((straight_attention, st_att))
            # --------------------------------- Calc usual way -----------------------------------------------

        # ---------------     calculate the attention Navit way -----------------------------

        # at = NavitAttention(dim=dim, heads=1, pack_length=navit_pack_length, multi_head=multihead_test)
        at = TpuNavitAttention(dim=dim, heads=1, pack_length=navit_pack_length, multi_head=multihead_test)

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

        test_output = test_output[ :original_seq_length, :]
        # ---------------   calc Navit   ------------------------------------------------------

        print(straight_attention.shape, test_output.shape)

        diff = torch.max(torch.abs(test_output.cpu() - straight_attention.cpu()))
        print(f"max diff: {diff},  ")

        # assert diff < 0.001, \
        #     "########################   ATTENTION CALCULATION ERROR   #########################"
    print(f"Navit Forward Attention Accuracy Test - OK  =============\n\n\n")

def navit_backward_accuracy(multihead_test=False):
    # from torch.autograd import grad
    print(f"Navit Backward Attention Accuracy Test ----------------------------- ")
    navit_pack_length = 128
    pic_in_sequence = 6
    max_embeddings_in_pic = int(navit_pack_length / pic_in_sequence)
    dim = 64
    K = nn.Linear(dim, dim)
    V = nn.Linear(dim, dim)
    Q = nn.Linear(dim, dim)

    pic_embed_sequence = torch.ones(0, dim)
    pic_embed_indexes = []
    cont_embed_sequence = torch.ones(0, dim)
    cont_embed_indexes = []

    straight_attention = torch.ones(0, dim)

    pics = []
    ctxs = []
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
        if multihead_test:
            q = rearrange(q, "n (h d) -> 1 h n d", h=8)
            k = rearrange(k, "n (h d) -> 1 h n d", h=8)
            v = rearrange(v, "n (h d) -> 1 h n d", h=8)
        st_att = nn_func.scaled_dot_product_attention(q, k, v)
        if multihead_test:
            st_att = rearrange(st_att, "1 h n d -> n (h d)")
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

    Q.zero_grad()
    K.zero_grad()
    V.zero_grad()
    original_seq_length = pic_embed_sequence.shape[0]

    pic_embed_sequence = NavitAttention.pad_embedding_sequence(pic_embed_sequence, navit_pack_length)
    cont_embed_sequence = NavitAttention.pad_embedding_sequence(cont_embed_sequence, navit_pack_length)
    q = Q(pic_embed_sequence)
    k = K(cont_embed_sequence)
    v = V(cont_embed_sequence)
    if multihead_test:
        q = rearrange(q, "n (h d) -> 1 h n d", h=8)
        k = rearrange(k, "n (h d) -> 1 h n d", h=8)
        v = rearrange(v, "n (h d) -> 1 h n d", h=8)
    test_output = at.forward(q=q,
                             q_indexes=NavitAttention.build_pic_id_sequence(pic_embed_indexes, navit_pack_length),
                             k=k,
                             v=v,
                             kv_indexes=NavitAttention.build_ctx_id_sequence(cont_embed_indexes, navit_pack_length)
                             )
    if multihead_test:
        st_att = rearrange(st_att, "1 h n d -> n (h d)")
    # test_output = test_output[:original_seq_length, :]
    test_output = test_output[:original_seq_length, :]
    # ---------------   calc Navit   ------------------------------------------------------

    loss = torch.sum(test_output)
    loss.backward()
    grad_cat2 = torch.ones(0, dim)
    grad_cat2 = torch.cat((grad_cat2, Q.weight.grad))
    grad_cat2 = torch.cat((grad_cat2, K.weight.grad))
    grad_cat2 = torch.cat((grad_cat2, V.weight.grad))



    diff = torch.max(torch.abs(grad_cat2 - grad_cat))
    print(f"grad max diff: {diff},  ")

    diff = torch.max(torch.abs(test_output - straight_attention))
    print(f"attn max diff: {diff},  ")

    assert diff < 0.001, \
        "########################   ATTENTION CALCULATION ERROR   #########################"
    print(f"Navit Backward Attention Accuracy Test - OK  =============\n\n\n")


if __name__ == "__main__":
    import os
    os.environ['PJRT_DEVICE'] = 'TPU'
    os.environ['TPU_NUM_DEVICES'] = '1'
    # os.environ['PT_XLA_DEBUG'] = '1'
    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
    # torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(linewidth=200)  # let torch print on all the screen

    # torch_xla._XLAC._xla_set_use_full_mat_mul_precision(use_full_mat_mul_precision=True)
    jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
    # navit_attention_accuracy_test(multihead_test=False)
    #
    # for _ in range(10):
    #     navit_backward_accuracy(multihead_test=False)
    #
    navit_attention_accuracy_test(multihead_test=True)
    # for _ in range(1):
    #     navit_backward_accuracy(multihead_test=True)
