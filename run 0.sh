#!/bin/bash
batch_sizes=($((64*10)) $((64*10)) $((64*5)) $((64*5)))
descriptor_fname=descriptors_cub_llama2_prompt_0_topk1_run_0
eval_fname=${descriptor_fname/descriptors/eval}
current_date=$(date +"%m%d")
eval_dir="eval_results $current_date"
model_sizes=('ViT-B/32' 'ViT-B/16' 'ViT-L/14' 'ViT-L/14@336px')
python3 main.py --batch_size ${batch_sizes[0]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[0]} --device cuda:0
python3 main.py --batch_size ${batch_sizes[1]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[1]} --device cuda:0
python3 main.py --batch_size ${batch_sizes[2]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[2]} --device cuda:0
python3 main.py --batch_size ${batch_sizes[3]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[3]} --device cuda:0

descriptor_fname=descriptors_cub_llama2_prompt_0_topk1_run_1
eval_fname=${descriptor_fname/descriptors/eval}
python3 main.py --batch_size ${batch_sizes[0]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[0]} --device cuda:0
python3 main.py --batch_size ${batch_sizes[1]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[1]} --device cuda:0
python3 main.py --batch_size ${batch_sizes[2]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[2]} --device cuda:0
python3 main.py --batch_size ${batch_sizes[3]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[3]} --device cuda:0