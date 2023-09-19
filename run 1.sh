#!/bin/bash
batch_sizes=($((64*10)) $((64*10)) $((64*5)) $((64*5)))
descriptor_fname=descriptors_cars_prompt0_run_1.json
clocktime=$(date +"%H:%M:%S")
eval_fname="${descriptor_fname/descriptors/eval}_$clocktime"
current_date=$(date +"%m_%d")
eval_dir="eval_results_$current_date"
model_sizes=('ViT-B/32' 'ViT-B/16' 'ViT-L/14' 'ViT-L/14@336px')
xxx=0
already_complete_descriptors=0
python3 main.py --batch_size ${batch_sizes[0]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[0]} --xxx $xxx --device cuda:1 --already_complete_descriptors $already_complete_descriptors
python3 main.py --batch_size ${batch_sizes[1]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[1]} --xxx $xxx --device cuda:1 --already_complete_descriptors $already_complete_descriptors
python3 main.py --batch_size ${batch_sizes[2]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[2]} --xxx $xxx --device cuda:1 --already_complete_descriptors $already_complete_descriptors
python3 main.py --batch_size ${batch_sizes[3]} --descriptor_fname $descriptor_fname --eval_fname $eval_fname --eval_dir $eval_dir --model_size ${model_sizes[3]} --xxx $xxx --device cuda:1 --already_complete_descriptors $already_complete_descriptors