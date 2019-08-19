# Set Transformer
python run.py --net=set_transformer --num_steps 2000 --exp_name set_transformer_run
python run.py --net=set_transformer --exp_name set_transformer_run --mode test

# Deepset
python run.py --net=deepset --num_steps 2000 --exp_name deepset_run
python run.py --net=deepset --exp_name set_transformer_run --mode test