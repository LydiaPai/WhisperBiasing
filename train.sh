# . /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate espnet
expdir=finetune_librispeech_lr0.0005_KB200_drop0.1
mkdir -p exp/${expdir}

python train.py \
    --train_json data/LibriSpeech/train_clean_100_full.json \
    --dev_json data/LibriSpeech/dev_clean_full.json \
    --lr 0.0005 \
    --batch_size 8 \
    --log_interval 20 \
    --nepochs 30 \
    --warmup_pct 0.0 \
    --decay_pct 0.2 \
    --expdir exp/${expdir} \
    --logfile exp/${expdir}/log.txt \
    --accumgrad 10 \
    --biasing \
    --biasinglist data/LibriSpeech/Blist/rareword_f15.txt \
    --dropentry 0.1 \
    --maxKBlen 200 \
    # --useGPT \
    # --GNNtype gcn2 \
    # --GNNdim 256 \
    # --loadfrom exp/finetune_librispeech_lr0.0005_KB200_drop0.1_GPThid/model.acc.best \
# CUDA_VISIBLE_DEVICES=1