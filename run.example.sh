#!/bin/bash

python train_mlm.py \
-b huggingface_model \
--ntrain 900000 \
--nval 200 \
--ckptrate 50000 \
--valrate 1000 \
--batch 32 \
--chunk 200 \
--lr '2e-5' \
--mlmprob 0.15 \
--seed 0 \
--data path/to/data \
--user wandb_user \
--project wandb_project_name \
--name wandb_run_name \
# --wholeword \
# --verbose \
# --droplast
