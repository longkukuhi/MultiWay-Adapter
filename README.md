# [MultiWay-Adapater: Adapting large-scale multi-modal models for scalable image-text retrieval]

Official PyTorch implementation and pretrained models of paper: MultiWay-Adapater: Adapting large-scale multi-modal models for scalable image-text retrieval. 


#### Run SimMIM with distributed training:
```bash
python run_beit3_finetuning.py --model ‘beit3_large_patch16_384’ --input_size 224 --task ‘coco_retrieval’ --batch_size 128 --layer_decay 0.65 --lr 2e-4 --epochs 30 --warmup_epochs 3 --drop_path 0.2 --sentencepiece_model ‘beit3.spm’ --data_path ‘path/to/your/dataset’ --output_dir 'coco_retrieval_output/' --log_dir '/coco_retrieval_log/' --weight_decay 0.05  --save_ckpt_freq 1 --finetune 'beit3_large_itc_patch16_224.pth' --num_max_bpe_tokens 64 --ffn_adapt --ffn_mode both --ffn_num 64 --ffn_adapter_scalar 0.6 --ffn_enhance
```
- `model` specifics the name of model we use in this experiments. 
- `log_dir` is the folder dir that stores the ouput log.
- `task`  specifics using coco or flickr30k dataset
- `data_path` is the folder dir that stores the datasets.
- `finetune` specifics the dir to pretrained weight of BEiT-3 model.
- `ffn_adapt` specifics to use the Multiway-Adapter
- `ffn_num` specifics the mid dimension of the NewKnowledge Extractor 
- `ffn_enhance` specifics to use the Alignment Enhancer



## Citation

If you find this repository useful, please consider citing works:
```
@article{long2023multiway,
  title={MultiWay-Adapater: Adapting large-scale multi-modal models for scalable image-text retrieval},
  author={Long, Zijun and Killick, George and McCreadie, Richard and Camarasa, Gerardo Aragon},
  journal={arXiv preprint arXiv:2309.01516},
  year={2023}
}
```


## Acknowledgement

This repository is built using the [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3) repository and the [timm](https://github.com/rwightman/pytorch-image-models) library.


