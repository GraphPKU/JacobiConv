import enum
import subprocess
import time


rep = 10
split = "dense"
dir = "ImgFilter"


def work(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python ImgFilter.py --repeat {rep} --optruns 400 --split {split} --device {gpu_id} --dataset {dataset}  --path {dir}/ --name {dataset} > {dir}/{dataset}{gpu_id}_{rep}_{split} 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def test(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python ImgFilter.py  --repeat 1 --device {gpu_id} --dataset {dataset} --split {split}   --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.test3 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def chebywork(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python ImgFilter.py --cheby --repeat {rep} --optruns 400 --split {split} --device {gpu_id} --dataset {dataset}  --path {dir}/ --name {dataset} > {dir}/{dataset}{gpu_id}_{rep}_{split} 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def jacobi(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python ImgFilter.py  --repeat 1 --test --dataset {dataset} --split {split}   --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.jacobi 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def power(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python ImgFilter.py --power  --repeat 1 --test --device {gpu_id} --dataset {dataset} --split {split}   --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.power 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def cheby(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python ImgFilter.py --cheby  --repeat 1 --test --device {gpu_id} --dataset {dataset} --split {split}   --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.cheby 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def legendre(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python ImgFilter.py --legendre  --repeat 1 --test --device {gpu_id} --dataset {dataset} --split {split}   --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.legendre 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def bern(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python ImgFilter.py --bern  --repeat 1 --test --device {gpu_id} --dataset {dataset} --split {split}   --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.bern 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)



def hinge(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python ImgFilter.py --hingeloss  --repeat 1 --test --device {gpu_id} --dataset {dataset} --split {split}   --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.hinge 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def getmodel(dg):
    dataset, gpu_id = dg
    cmd = f"nohup python ImgFilter.py --getmodel  --repeat 1 --test --device {gpu_id} --dataset {dataset} --split {split}   --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.getmodel 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def wait():
    while True:
        ret = subprocess.check_output("nvidia-smi -q -d Memory | grep  Used",
                                      shell=True)
        sel = [
            _ for _ in ret.split()
            if b"U" not in _ and b":" not in _ and b"M" not in _
        ]
        load = [int(i) for i in sel]
        for i in range(len(load) // 2):
            if sum(load[2 * i:2 * i + 2]) < 5000:
                return i
        time.sleep(30)


for fn in [jacobi]:
    for ds in ['low', "high", 'rejection', 'band', 'comb']:
        dev = 1
        fn((ds, dev))
        break