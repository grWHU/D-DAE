# D-DAE
official PyTorch implementation of D-DAE
## Installation

The code successfully runs on Python 3.7 and PyTorch 1.6. Required packages can be installed by:
```bash
pip install -r requirements.txt
```

## Defense Detection

This repo provides an implementation of detecting Trojans in machine learning models as introduced [here](https://arxiv.org/abs/1910.03137).

### Training Shadow Models and Target Models

An example of running on the MNIST task:

```bash
python train_shadow.py --task mnist
```

### Training and Evaluating the Meta-Classifier

`run_meta.py` trains and evaluates the meta-classifier. 

An example of training the meta-classifier with a vicim protected by ReverseSigmoid on the MNIST task and evaluating on modification attack:

```bash
python detection.py --task mnist --defense MAD --target ReverseSigmoid
```

## Defense Disruption
### Datasets
You will need six datasets to perform all experiments in the paper, 
all extracted into the `data/` directory. 
Some of the datasets (e.g., MNIST, CIFAR) used in the paper is automatically downloaded when running the experiments for the first time.
However, the rest (e.g., Caltech256, CUB200, ImageNet) need to be downloaded manually -- please follow the instructions [here](https://github.com/tribhuvanesh/knockoffnets#datasets).  

### Victim Models

You can train victim models deployed with various kinds of defense mechanisms.

For victim models with no defense, using the command below:
 
```bash
# Format:
$ python online/victim/train.py DS_NAME ARCH -d DEV_ID \
        -o models/victim/VIC_DIR -e EPOCHS --pretrained
# where DS_NAME = {MNIST, FashionMNIST, CIFAR10, GTSRB, ImageNette}, ARCH = {lenet, vgg16_bn, resnet34, ...}
# if the machine contains multiple GPUs, DEV_ID specifies which GPU to use

# More details:
$ python online/victim/train.py --help

# Example (MNIST):
$ python online/victim/train.py MNIST lenet -d 1 \
        -o models/victim/MNIST-lenet-train-nodefense -e 100 --log-interval 25
```

We consider 5 kinds of defenses, PP(Prediction Poisoning) and RS(Reverse Sigmoid) are now available.

Execute experiments with the following setting:
* Defense = MAD / ReverseSigmoid
* Attack = Knockoff / Active Thief / JBDA
* Dataset (Victim Model) = MNIST / FashionMNIST / CIFAR-10 / GTSRB / ImageNette
* Queryset = EMNISTLetter / EMNIST / CIFAR-10 / GTSRB / ImageNette

These above parameters can be changed by simply substituting the variables with the one you want


### Attack Models


### OptionA. KnockoffNets

#### Step 1: Setting up experimental variables
If the victim model is protected by PP, then:
```bash
### If you have multiple GPUs on the machine, use this to select the specific GPU
dev_id=0
### Metric for perturbation ball dist(y, y'). Supported = L1, L2, KL
y_dist=l1
### Perturbation norm
eps=0.5/0.99/1.1
### p_v = victim model dataset
p_v=MNIST
### f_v = architecture of victim model
f_v=lenet 
### queryset = p_a = image pool of the attacker 
queryset=EMNISTLetters
### Path to victim model's directory (the one downloded earlier)
vic_dir=models/victim/${p_v}-${f_v}-train-nodefense;
### No. of images queried by the attacker. With 60k, attacker obtains 99.05% test accuracy on MNIST at eps=0.0.
budget=500,5000,20000,60000 
### Initialization to the defender's surrogate model. 'scratch' refers to random initialization.
proxystate=scratch;
### Path to surrogate model
proxydir=models/victim/${p_v}-${f_v}-train-nodefense-${proxystate}-advproxy
### Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}-mad_${ydist}-eps${eps}-${queryset}-B${budget}-proxy_${proxystate}-random
### Defense strategy
strat=mad
### Parameters to defense strategy, provided as a key:value pair string. 
defense_args="epsilon:${eps},objmax:True,ydist:${ydist},model_adv_proxy:${proxydir},out_path:${out_dir}"
### Batch size of queries to process for the attacker
batch_size=1
```

If the victim model is protected by RS, then:

```bash
### If you have multiple GPUs on the machine, use this to select the specific GPU
dev_id=0
### Perturbation norm
beta=0.1/0.3/0.5/0/8
gamma=0.2(MNIST & ImageNette)/0.4(FashionMNIST)/0.1(CIFAR-10 & GTSRB)
### p_v = victim model dataset
p_v=MNIST
### f_v = architecture of victim model
f_v=lenet 
### queryset = p_a = image pool of the attacker 
queryset=EMNISTLetters
### Path to victim model's directory (the one downloded earlier)
vic_dir=models/victim/${p_v}-${f_v}-train-nodefense;
### No. of images queried by the attacker. With 60k, attacker obtains 99.05% test accuracy on MNIST at eps=0.0.
budget=500,5000,20000,60000 
### Output path to attacker's model
out_dir=models/final_bb_dist/${p_v}-${f_v}-reverse_sigmoid-beta${beta}-gamma${gamma}-${queryset}-B${budget}-proxy_scratch-random
### Defense strategy
strat=reverse_sigmoid
### Parameters to defense strategy, provided as a key:value pair string. 
defense_args="beta:${beta},gamma:${gamma},out_path:${out_dir}"
### Batch size of queries to process for the attacker
batch_size=1
```

#### Step2: Simulate Attacker Interactions
The command below constructs the attackers transfer set, i.e., images and their corresponding pseudo-labels (perturbed posteriors) obtained by querying the defended blackbox. 
The defense is configured by `strat` and `defense_args` variables:
```bash
$ python defenses/adversary/transfer.py random ${vic_dir} ${strat} ${defense_args} \
    --out_dir ${out_dir} \
    --batch_size ${batch_size} \
    -d ${dev_id} \
    --queryset ${queryset} \
    --budget ${budget}
  
# Example (MNIST with MAD defense)
$ python defenses/adversary/transfer.py random models/victim/MNIST-lenet-train-nodefense mad epsilon:0.5,model_adv_proxy:models/victim/MNIST-lenet-train-nodefense-scratch-advproxy,out_path:models/final_bb_dist/MNIST-lenet-mad_l1-eps0.8-EMNISTLetters-B60000-proxy_scratch-random \
    --out_dir models/final_bb_dist/MNIST-lenet-mad_l1-eps0.8-EMNISTLetters-B60000-proxy_scratch-random \
    --batch_size 1 \
    -d 0 \
    --queryset EMNISTLetters \
    --budget 60000
    
# Example (MNIST with RS defense)
$ python defenses/adversary/transfer.py random models/victim/MNIST-lenet-train-nodefense reverse_sigmoid beta:0.1,gamma:0.2,out_path:models/final_bb_dist/MNIST-lenet-reverse_sigmoid-beta0.8-gamma0.2-EMNISTLetters-B60000-proxy_scratch-random \
    --out_dir models/final_bb_dist/MNIST-lenet-reverse_sigmoid-beta0.8-gamma0.2-EMNISTLetters-B60000-proxy_scratch-random \
    --batch_size 1 \
    -d 0 \
    --queryset EMNISTLetters \
    --budget 60000

```

#### Step 3: Train + Evaluate Attacker

After the transfer set (i.e., attacker's training set) is constructed, the command below trains multiple attack models for various choices of sizes of transfer sets (specified by `budgets`).
During training, the model is simulatenously evaluated during each epoch. 

```bash
$ python online/adversary/train.py ${out_dir} ${f_v} ${p_v} \
    --budget 500,5000,20000,60000 \
    --log-interval 500 \
    --epochs 50 \
    -d ${dev_id}
    
# Example (MNIST with RS defense)    
$ python online/adversary/train.py models/final_bb_dist/MNIST-lenet-mad_l1-eps0.5-EMNISTLetters-B60000-proxy_scratch-random lenet MNIST \
    --budget 500,5000,20000,60000 \
    --log-interval 500 \
    --epochs 50 \
    -d 0    
    
# Example (MNIST with RS defense)    
$ python online/adversary/train.py models/final_bb_dist/MNIST-lenet-reverse_sigmoid-beta0.1-gamma0.2-EMNISTLetters-B60000-proxy_scratch-random lenet MNIST \
    --budget 500,5000,20000,60000 \
    --log-interval 500 \
    --epochs 50 \
    -d 0

``` 

#### Step 4: Train + Evaluate D-DAE version of Attacker

The command is the same compared with Step3, you have to make some changes:

First, in `/knockoff/adversary/train.py` `Line 214`
```bash
model_utils.train_model(model, transferset, model_dir, testset=testset, criterion_train=criterion_train,
                                checkpoint_suffix=checkpoint_suffix, device=device, restored=True, task=args.testdataset,
                                optimizer=optimizer, **params)
```
Note: change  `restored=False` to `restored=True`.

Second, in `/knockoff/utils/model.py` `Line 104 - Line 127`
```bash
    if task == 'MNIST':
        Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting("mnist")
        generator_path = ${Generator}
        meta_path = ${Meta-classifer}
# Generator stored in /generator/${task}${defense}
# Meta-classifier stored in /meta/${defense}${task}
# If you don't have the specific kind of Generator or Meta-Classifier, please text to me for help !!!    
```

##### Very Important Note:

In Step 4, You have to ensure the attack success rate is higher than the success rate in Step 3.
To achieve this, you can adjust `--lr`, `--momentum`, `--lr-gamma`, set the budget small to see quick feedbacks.

Specifically,
```bash
# Example (MNIST with RS defense)    
$ python online/adversary/train.py models/final_bb_dist/MNIST-lenet-mad_l1-eps0.5-EMNISTLetters-B60000-proxy_scratch-random lenet MNIST \
    --budget 500 \
    --log-interval 500 \
    --epochs 50 \
    --lr 0.1
    --momentum 0.5 \
    -d 0 
```

### OptionB. JBDA

#### Step 1: Setting up experimental variables

If the victim model is protected by PP, then:
```bash
### If you have multiple GPUs on the machine, use this to select the specific GPU
dev_id=0
### Metric for perturbation ball dist(y, y'). Supported = L1, L2, KL
y_dist=l1
### Perturbation norm
eps=0.5/0.99/1.1
### p_v = victim model dataset
p_v=MNIST
### f_v = architecture of victim model
f_v=lenet 
### queryset = p_a = image pool of the attacker 
queryset=EMNISTLetters
### Path to victim model's directory (the one downloded earlier)
vic_dir=models/victim/${p_v}-${f_v}-train-nodefense;
### No. of images queried by the attacker. With 60k, attacker obtains 99.05% test accuracy on MNIST at eps=0.0.
budget=500,5000,20000,60000 
### Initialization to the defender's surrogate model. 'scratch' refers to random initialization.
proxystate=scratch
### Path to surrogate model
proxydir=models/victim/${p_v}-${f_v}-train-nodefense-${proxystate}-advproxy
### Output path to attacker's model
out_dir=models/adversary/${p_v}-${f_v}-jbda-${strat}_${y_dist}-eps${eps}-${queryset}-B${budget}
### Defense strategy
strat=mad
### Parameters to defense strategy, provided as a key:value pair string. 
defense_args="epsilon:${eps},objmax:True,ydist:${ydist},model_adv_proxy:${proxydir},out_path:${out_dir}"
### Batch size of queries to process for the attacker
batch_size=1
```
If the victim model is protected by RS, then is basically the same.

#### Step2: Simulate Attacker Interactions
Totally same as Step 2 in Option A.

#### Step3: Train + Evaluate Attacker
```bash
$ python defenses/adversary/jacobian.py jbda ${victim_dir} ${strat} ${defense_args} \
    --model_adv ${f_v} \
    --out_dir ${out_dir} 
    --testset ${p_v} \
    --budget 500,5000,20000,60000 \
    --queryset ${queryset} \
    -d 1
    
# Example (PP)
$ python defenses/adversary/jacobian.py jbda models/victim/MNIST-lenet-train-nodefense mad epsilon:0.5,model_adv_proxy:models/victim/MNIST-lenet-train-nodefense-scratch-advproxy,out_path:models/adversary/MNIST-lenet-jbda-mad_l1-eps0.5-EMNISTLetters-B60000 \
    --model_adv lenet \
    --out_dir models/adversary/MNIST-lenet-jbda-mad_l1-eps0.5-EMNISTLetters-B60000 \
    --testset MNIST \
    --budget 500,5000,20000,60000 \
    --queryset EMNISTLetters \
    -d 1
    
# Example (RS)
$ python defenses/adversary/jacobian.py jbda models/victim/MNIST-lenet-train-nodefense reverse_sigmoid beta:0.1,gamma:0.2,out_path:models/adversary/MNIST-lenet-jbda-reverse_sigmoid-beta0.1-gamma0.2-EMNISTLetters-B60000 \
    --model_adv lenet \
    --out_dir models/adversary/MNIST-lenet-jbda-reverse_sigmoid-beta0.1-gamma0.2-EMNISTLetters-B60000 \
    --testset MNIST \
    --budget 500,5000,20000,60000 \
    --queryset EMNISTLetters \
    -d 1

```
#### Step 4: Train + Evaluate D-DAE version of Attacker

The command is the same compared with Step3, you have to make some changes:

First, in `/defenses/adversary/jacobian.py` `Line 157`
```bash
model_adv = model_utils.train_model(model_adv, self.D, self.out_dir, num_workers=10,
                          checkpoint_suffix='.{}'.format(self.blackbox.call_count),
                          device=self.device, restored=False, task=self.testset_name,
                          epochs=self.final_train_epochs,
                          log_interval=500, lr=0.01, momentum=0.9, batch_size=self.batch_size,
                          lr_gamma=0.1, testset=self.testset_name,
                          criterion_train=model_utils.soft_cross_entropy)
```
Note: change  `restored=False` to `restored=True`.

Second, in `/knockoff/utils/model.py` `Line 104 - Line 127`
```bash
    if task == 'MNIST':
        Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting("mnist")
        generator_path = ${Generator}
        meta_path = ${Meta-classifer}
# Generator stored in /generator/${task}${defense}
# Meta-classifier stored in /meta/${defense}${task}
# If you don't have the specific kind of Generator or Meta-Classifier, please text to me for help !!!    
```

##### Very Important Note:

In Step 4, You have to ensure the attack success rate is higher than the success rate in Step 3.
To achieve this, you can adjust `--lr`, `--momentum`, `--lr-gamma`, set the budget small to see quick feedbacks.

### OptionC. ActiveThief
Same setting as the above two, only provide examples below:
```bash
# for PP
$ python active.py kcenter mad epsilon:0.5,model_adv_proxy:models/victim/MNIST-lenet-train-nodefense-scratch-advproxy,out_path:models/adversary/MNIST-lenet-kcenter-mad-eps0.5 EMNISTLetters models/victim/MNIST-lenet-train-nodefense models/adversary/MNIST-lenet-kcenter lenet MNIST \
    -e 200 \
    --metric l1 \
    -d 0 \
    --batch-size 256 \
    --initial-size 500 \
    --budget-per-iter 500 \
    --iterations 29
    
# for RS
$ python active.py kcenter reverse_sigmoid beta:0.1,gamma:0.2,out_path:models/adversary/MNIST-lenet-kcenter-reverse_sigmoid-beta0.1-gamma0.2 EMNISTLetters models/victim/MNIST-lenet-train-nodefense models/adversary/MNIST-lenet-kcenter lenet MNIST \
    -e 200 \
    --metric l1 \
    -d 0 \
    --batch-size 256 \
    --initial-size 500 \
    --budget-per-iter 500 \
    --iterations 29
```
Budget = innitial-size + budget-per-iter * iterations.
