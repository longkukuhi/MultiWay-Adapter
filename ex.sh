python  run_beit3_finetuning.py --model 'beit3_base_patch16_384' \
--input_size 384 --task 'coco_retrieval' \
--finetune '../ckpt/beit3/beit3_base_patch16_384_coco_retrieval.pth' \
--layer_decay 0.65 --lr 5e-4 --epochs 10 --warmup_epochs 2 \
--drop_path 0.2 --sentencepiece_model 'beit3.spm' \
--data_path '../datasets/coco/' --output_dir './coco_retrieval_output/' \
--log_dir './coco_retrieval_log/' --weight_decay 0.05 --seed 42 \
 --save_ckpt_freq 1 --num_workers 16 --num_max_bpe_tokens 64 --batch_size 64 \
--eval_batch_size 128  --ffn_adapt --ffn_mode 'both' --enable_wandb \
--ffn_num 64 --ffn_enhance --ffn_adapter_scalar 1