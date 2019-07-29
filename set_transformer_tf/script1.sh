for i in {0..4}
do
    python run.py --enc ff --dec ff  --exp_name run$i --gpu 4
    python run.py --enc ff --dec ff  --exp_name run$i --gpu 4 --mode test

    python run.py --enc ff --dec sab --exp_name run$i --gpu 4
    python run.py --enc ff --dec sab --exp_name run$i --gpu 4 --mode test

    python run.py --enc sab --dec ff --exp_name run$i --gpu 4

    python run.py --N_min 1000 --N_max 5000 --K 6 --B 5 --exp_name run$i-large --gpu 5
    python run.py --N_min 1000 --N_max 5000 --K 6 --B 5 --exp_name run$i-large --gpu 5 --mode test

    python run.py --N_min 1000 --N_max 5000 --K 6 --B 5 --enc perm_eq_mean --exp_name run$i-large --gpu 5
    python run.py --N_min 1000 --N_max 5000 --K 6 --B 5 --enc perm_eq_mean --exp_name run$i-large --gpu 5 --mode test

    python run.py --enc sab --n_ind 32 --dec sab --N_min 1000 --N_max 5000 --K 6 --B 5 --exp_name run$i-large --gpu 6
    python run.py --enc sab --n_ind 32 --dec sab --N_min 1000 --N_max 5000 --K 6 --B 5 --exp_name run$i-large --gpu 6 --mode test

    python run.py --enc perm_eq_max --dec ff --exp_name run$i --gpu 7
    python run.py --enc perm_eq_max --dec ff --exp_name run$i --gpu 7 --mode test

    python run.py --enc ff --dec dotprod --exp_name run$i --gpu 7
    python run.py --enc ff --dec dotprod --exp_name run$i --gpu 7 --mode test

done
