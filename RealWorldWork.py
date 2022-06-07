import subprocess
import time

rep = 10
split = "dense"
dir = "RealWorld"


def work(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --repeat {rep} --optruns 400 --split {split} --dataset {dataset}   --path {dir}/ --name {dataset} > {dir}/{dataset}{gpu_id}_{rep}_{split} 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def jacobi(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.jacobi 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def power(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --power --test --repeat 10 --dataset {dataset} --split {split} --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.power 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def cheby(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --cheby --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.cheby 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def legendre(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --legendre --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.legendre 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def bern(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --bern --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.bern 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def unifilter(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --sole --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.unifilter 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def noPCD(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --fixalpha --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.nopcd 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def getmodel(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --getmodel --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.getmodel 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def power_noPCD(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --fixalpha --power --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.power 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def multilayer(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --multilayer --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.multilayer 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)

def cheby_noPCD(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --fixalpha --cheby --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.cheby 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def legendre_noPCD(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --legendre --fixalpha --test --repeat 10  --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.legendre 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def jacobi_noPCD_sel_ab(dg):
    dataset, gpu_id = dg
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python RealWorld.py --fixalpha --test --repeat 10 --dataset {dataset} --split {split}    --path {dir}/ --name {dataset}  >  {dir}/{dataset}{gpu_id}.jacobi_sel_ab 2>&1 &"
    print(cmd, flush=True)
    subprocess.call(cmd, shell=True)


def wait():
    while True:
        time.sleep(15)
        ret = subprocess.check_output("nvidia-smi -q -d Memory | grep  Used",
                                      shell=True)
        sel = [
            _ for _ in ret.split()
            if b"U" not in _ and b":" not in _ and b"M" not in _
        ]
        used = [int(i) for i in sel]
        for j in range(4):
            if (used[2*j] < 9000):
                return j


for fn in [jacobi]:
    for ds in ['pubmed', 'computers', "squirrel", "photo", "chameleon", "film", "cora", "citeseer", "texas", "cornell"]:
        gpu = wait()
        fn((ds, gpu))
