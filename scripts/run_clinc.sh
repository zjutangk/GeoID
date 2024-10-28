#!/usr/bin/env bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=3,4
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
for d in 'clinc'
do
	for l in  0.1
	do
        	for k in 0.5
		do
			for s in 1
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
				--feat_dim   256\
				--bert_model "./pretrained_models/base"\
				>>exper/0204_clinc_0.5_kcr.txt
    			done
		done
	done 
done