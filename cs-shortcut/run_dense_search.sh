#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run_dense_search.py --data_file test_fused_simple_ZSL_post.json --read_by GPT_rewrite

# CUDA_VISIBLE_DEVICES=1 python run_dense_search.py --data_file test_fused_larity_ZSL_post.json --read_by GPT_rewrite

# CUDA_VISIBLE_DEVICES=1 python run_dense_search.py --data_file test_fused_orrect_ZSL_post.json --read_by GPT_rewrite
