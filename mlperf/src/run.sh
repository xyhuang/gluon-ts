NOW=`date "+%F-%T"`

python train_and_eval.py \
  --device gpu \
  --num-epochs-per-iteration 5 \
  --num-iterations 2 \
  --result-file results-${NOW}.csv \
  --log-file log-${NOW}.log
