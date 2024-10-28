#!/usr/bin/env bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=2,3
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export DISPLAY=:10.0
for d in 'banking'
do
	for l in  0.1
	do
        	for k in 0.25
		do
			for s in 0 
        		do
	    		python geoid.py \
				--data_dir data \
				--dataset $d \
				--known_cls_ratio $k \
				--labeled_ratio $l \
				--seed $s \
				--lr '1e-3' \
				--save_results_path 'clnn_outputs' \
				--view_strategy 'rtr' \
				--update_per_epoch 5 \
				--topk 64 \
				--num_train_epochs 200 \
				--num_warm_epochs  20\
				--train_batch_size 128\
				--feat_dim   128\
				--num_pretrain_epochs 50\
				--bert_model "./pretrained_models/base"\
				>> exper/214banking_testtime.txt
    			done
		done
	done 
done
