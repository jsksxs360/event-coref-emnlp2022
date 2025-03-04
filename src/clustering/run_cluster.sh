export OUTPUT_DIR=./TEMP/

python3 run_cluster.py \
    --output_dir=$OUTPUT_DIR \
    --test_golden_filepath=../../data/test.json \
    --test_pred_filepath=../../data/XXX_weights.bin_test_pred_corefs.json \
    --golden_conll_filename=gold_test.conll \
    --pred_conll_filename=pred_test.conll \
    --do_evaluate \
    # --do_rescore \
    # --rescore_reward=0.5 \
    # --rescore_penalty=0.5