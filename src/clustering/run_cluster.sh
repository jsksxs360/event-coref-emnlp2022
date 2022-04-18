export OUTPUT_DIR=./TD_CE_CR_CE/

python3 run_cluster.py \
    --output_dir=$OUTPUT_DIR \
    --test_golden_filepath=../../data/test.json \
    --test_pred_filepath=../../data/epoch_12_dev_f1_67.1188_weights.bin_test_pred_corefs.json \
    --golden_conll_filename=gold_test.conll \
    --pred_conll_filename=pred_test.conll \
    --do_evaluate