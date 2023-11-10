#!/usr/bin/env bash

set -ex


REGEX_TIER1='^(BERT_pytorch|cm3leon_generate|DALLE2_pytorch|dlrm|hf_GPT2|hf_GPT2_large|GPT_3|hf_T5|hf_T5_base|hf_T5_generate|hf_T5_large|llama_v2_7b_16h|stable_diffusion_xl)$'
REGEX_TIER2='^(alexnet|attention_is_all_you_need_pytorch|Background_Matting|basic_gnn_gcn|basic_gnn_gin|basic_gnn_sage|dcgan|densenet121|detectron2_fasterrcnn_r_101_c4|detectron2_fasterrcnn_r_101_dc5|detectron2_fasterrcnn_r_101_fpn|detectron2_fasterrcnn_r_50_c4|detectron2_fasterrcnn_r_50_dc5|detectron2_fasterrcnn_r_50_fpn|detectron2_fcos_r_50_fpn|detectron2_maskrcnn|detectron2_maskrcnn_r_101_c4|detectron2_maskrcnn_r_101_fpn|detectron2_maskrcnn_r_50_c4|detectron2_maskrcnn_r_50_fpn|fastNLP_Bert|functorch_dp_cifar10|hf_Albert|hf_Bart|hf_Bert|hf_Bert_large|llama)$'
REGEX_TIER3='^(doctr_det_predictor|doctr_reco_predictor|drq|functorch_maml_omniglot)$'
REGEX_TIER4='^(hf_BigBird|hf_clip|hf_DistilBert|hf_Longformer|hf_Reformer|hf_Whisper|LearningToPaint|lennard_jones|maml|maml_omniglot|mnasnet1_0|mobilenet_v2|mobilenet_v2_quantized_qat|mobilenet_v3_large|nanogpt|nvidia_deeprecommender|opacus_cifar10|phi_1_5|phlippe_densenet|phlippe_resnet|pyhpc_equation_of_state|pyhpc_isoneutral_mixing|pyhpc_turbulent_kinetic_energy|pytorch_CycleGAN_and_pix2pix|pytorch_stargan|pytorch_unet|resnet152|resnet18|resnet50|resnet50_quantized_qat|resnext50_32x4d|sam|shufflenet_v2_x1_0|simple_gpt|simple_gpt_tp_manual|soft_actor_critic|speech_transformer|squeezenet1_1|Super_SloMo|tacotron2|timm_efficientdet|timm_efficientnet|timm_nfnet|timm_regnet|timm_resnest|timm_vision_transformer|timm_vision_transformer_large|timm_vovnet|tts_angular|vgg16|vision_maskrcnn|yolov3)$'


# REGEX_TIER1='^(alexnet)$'
# REGEX_TIER2='^(basic_gnn_gcn)$'
# REGEX_TIER3='^()$'
# REGEX_TIER4='^()$'




FILTER='^$'
while [[ -n "$1" ]]; do
    case $1 in
        -t | --tier)  
            TIER=$2
            if [[ "$TIER" == 1 ]]; then 
                FILTER="$FILTER|$REGEX_TIER1"
            elif [[ "$TIER" == 2 ]]; then 
                FILTER="$FILTER|$REGEX_TIER2"
            elif [[ "$TIER" == 3 ]]; then 
                FILTER="$FILTER|$REGEX_TIER3"
            elif [[ "$TIER" == 4 ]]; then 
                FILTER="$FILTER|$REGEX_TIER4"
            fi
            shift 2 
            ;;
        --) 
            shift 
            break
            ;;
        *)  
            echo "unk $1" 
            shift   
            ;;
    esac
done


echo "Running benchmarks with filter $FILTER"





# Run one benchmark 
# python benchmark/experiment_runner.py --dynamo=openxla_eval --xla=PJRT --test=eval --filter=basic_gnn_gcn$ --suite-name=torchbench --accelerator=cuda --progress-bar --output-dirname=/tmp/output --repeat=5 --print-subprocess --no-resume 





mkdir -p experiment_results

python benchmark/experiment_runner.py \
    --dynamo=inductor --dynamo=openxla_eval --dynamo=openxla \
    --xla=None --xla=PJRT \
    --test=eval --test=train \
    --filter="$FILTER" \
    --suite-name=torchbench \
    --accelerator=cuda \
    --output-dirname=experiment_results \
    --repeat=5 \
    --print-subprocess \
    --no-resume > experiment_results/stdout.txt 2> experiment_results/stderr.txt

python3 benchmark/result_analyzer.py \
    --output-dirname=experiment_results  \
    --database=experiment_results/database.csv



