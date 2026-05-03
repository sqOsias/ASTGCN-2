python scripts/compare_model_complexity.py \
  --config configurations/PEMS04.conf \
  --device auto \
  --batch_size 1 \
  --warmup 20 \
  --runs 100 \
  --repeats 10 \
  --seed 42