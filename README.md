# Social Media Promotion via Interaction Data Poisoning

This project is for the paper "Social Media Promotion via Interaction Data Poisoning".

The code was developed on Python 3.6 and tensorflow 1.15.0.

## Usage
### prepare Weibo data
```
mkdir data/weibo/fast
```
Put the data in this folder.

### run main.py
Generate poisoning data of IPRO.
```
python main.py --data weibo --model mlp1 --data_size 0.4 --poison 0 --train 1 --method inf --cal_inf --cal_cand
```
Evaluate IPRO
```
python main.py --data weibo --model mlp --data_size 0.4 --poison 1 --train 1 --method inf
```