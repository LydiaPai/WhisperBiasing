expdir=$1
expdir="/WhisperBiasing/exp/finetune_librispeech_lr0.0005_KB200_drop0.1/decode_no_lm_b50_KB1000_clean_50best_biasing_beam"
/espnet/tools/sctk/bin/sclite -r $expdir/ref.wrd.trn trn -h $expdir/hyp.wrd.trn trn -i rm -o all stdout > $expdir/results.txt
