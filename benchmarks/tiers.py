from typing import List


def append_filter_by_tier(filter_list: List[str], filter_by_tier: List[int]):
  _FILTER_BY_TIER = {
      1:
          r"^(BERT_pytorch|cm3leon_generate|DALLE2_pytorch|dlrm|hf_GPT2|hf_GPT2_large|GPT_3|hf_T5|hf_T5_base|hf_T5_generate|hf_T5_large|llama_v2_7b_16h|stable_diffusion_xl)$",
      2:
          r"^(alexnet|attention_is_all_you_need_pytorch|Background_Matting|basic_gnn_gcn|basic_gnn_gin|basic_gnn_sage|dcgan|densenet121|detectron2_fasterrcnn_r_101_c4|detectron2_fasterrcnn_r_101_dc5|detectron2_fasterrcnn_r_101_fpn|detectron2_fasterrcnn_r_50_c4|detectron2_fasterrcnn_r_50_dc5|detectron2_fasterrcnn_r_50_fpn|detectron2_fcos_r_50_fpn|detectron2_maskrcnn|detectron2_maskrcnn_r_101_c4|detectron2_maskrcnn_r_101_fpn|detectron2_maskrcnn_r_50_c4|detectron2_maskrcnn_r_50_fpn|fastNLP_Bert|functorch_dp_cifar10|hf_Albert|hf_Bart|hf_Bert|hf_Bert_large|llama)$",
      3:
          r"^(doctr_det_predictor|doctr_reco_predictor|drq|functorch_maml_omniglot)$",
  }
  for tier in filter_by_tier:
    if tier not in _FILTER_BY_TIER:
      continue
    filter_list.append(_FILTER_BY_TIER[tier])
