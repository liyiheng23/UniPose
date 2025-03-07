# stage 1 仅训练lm_head + text embd
# stage 2 load stage 1, only train mm_projector
# stage 3 load stage 2, train lm_head + text embd + mm_projector, 但是stage2仅保存的mm_projector的参数
# 所以，this file is to merge stage1 and stage2 non_lora_trainables.bin