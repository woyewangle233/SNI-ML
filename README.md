# SNI-ML

Code for the paper Research on Twitter Summary Based on  Social Network Information and Multimodal Learning.


Requirements and Installation：

PyTorch version >= 1.4.0
Python version >= 3.6
Download Data

Preprocess：

python process_hierarchical_sent_doc.py --source-lang src --target-lang tgt \
  --trainpref ./data/2000-300/train --validpref ./data/2000-300/valid --testpref ./data/2000-300/test \
  --destdir multi-news-2000-300-train --joined-dictionary --nwordssrc 50000 --workers 10
python process_hierarchical_sent_doc_copy.py --source-lang src --target-lang tgt \
  --testpref ./data/2000-300/test --destdir multi-news-2000-300-copy --workers 10 \
  --srcdict multi-news-2000-300-train/dict.src.txt --tgtdict multi-news-2000-300-train/dict.tgt.txt \
  --dataset-impl raw
  
Train:

 python train.py multi-news-2000-300-train -a hierarchical_transformer_medium \
--optimizer adam --lr 0.0001 -s src -t tgt --dropout 0.1 --max-tokens 2000   \
--share-decoder-input-output-embed   --task multi_loss_sent_word --adam-betas '(0.9, 0.98)' \
--save-dir checkpoints/hierarchical_transformer-2000-300 --share-all-embeddings  \
--lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 --criterion multi_loss_doc_sent_word \
--ddp-backend no_c10d --num-workers 2 \
--update-freq 13 --encoder-normalize-before --decoder-normalize-before --sent-weight 2

Test-abstractive:

python generate_for_hie.py multi-news-2000-300-copy --task multi_loss_sent_word \

Citation:
