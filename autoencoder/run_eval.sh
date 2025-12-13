CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12345 \
    evaluate_tokenizer.py --config_path pre-trained/autoencoder/svg_autoencoder_P_1024.yaml \
    --ckpt_path pre-trained/autoencoder/svg_autoencoder-P.ckpt \
    --model_type "dinov3" --data_path YourPath/ImageNet-1k/data/val_images --output_path eval \
    --ref_path ref_images ;\
