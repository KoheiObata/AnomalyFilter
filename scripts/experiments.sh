gpu=0
for i in 0
do
  echo "${i}"
  python main.py --gpu $gpu --dataset anomaly_archive --seed $i
  # python main.py --gpu $gpu --dataset iops --seed $i
  # python main.py --gpu $gpu --dataset yahoo_real --seed $i
  # python main.py --gpu $gpu --dataset yahoo_bench --seed $i
  # python main.py --gpu $gpu --dataset smd --seed $i --residual-layers 4 --residual-channels 32
done