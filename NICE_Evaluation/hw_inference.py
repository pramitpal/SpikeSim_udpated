import torch.optim as optim
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from hw_models_new import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import os.path
import os
import numpy as np
import torch.backends.cudnn as cudnn
from utills import *
import time
import sys

cudnn.benchmark = True
cudnn.deterministic = True

#--------------------------------------------------
# Parse input arguments
#--------------------------------------------------
parser = argparse.ArgumentParser(description='SNN trained with BNTT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed',                  default=0,        type=int,   help='Random seed')
parser.add_argument('--num_steps',             default=5,    type=int, help='Number of time-step')
parser.add_argument('--batch_size',            default=4,       type=int,   help='Batch size')
parser.add_argument('--ADC_precision',            default=4,       type=int,   help='Crossbar ADC Precision')
parser.add_argument('--quant',            default=4,       type=int,   help='Weight Quantization')
parser.add_argument('--xbar_size',            default=64,       type=int,   help='Crossbar Size')
parser.add_argument('--lr',                    default=0.1,   type=float, help='Learning rate')
parser.add_argument('--leak_mem',              default=0.99,   type=float, help='Leak_mem')
parser.add_argument('--arch',              default='vgg9',   type=str, help='Dataset [vgg9, vgg11]')
parser.add_argument('--dataset',              default='cifar10',   type=str, help='Dataset [cifar10, cifar100]')
parser.add_argument('--num_epochs',            default=120,       type=int,   help='Number of epochs')
parser.add_argument('--num_workers',           default=1, type=int, help='number of workers')
parser.add_argument('--b_size',           default=4, type=int, help='number of workers')
parser.add_argument('--bntt', default=True)

global args
args = parser.parse_args()


#--------------------------------------------------
# Initialize directories
#--------------------------------------------------
log_dir = 'modelsave'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

hw_outputs_dir = './hw_outputs'
if not os.path.isdir(hw_outputs_dir):
    os.makedirs(hw_outputs_dir)

hw_outs_new_dir = './hw_outs_new'
if not os.path.isdir(hw_outs_new_dir):
    os.makedirs(hw_outs_new_dir)


#--------------------------------------------------
# Initialize seed
#--------------------------------------------------
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#--------------------------------------------------
# SNN configuration parameters
#--------------------------------------------------
leak_mem = args.leak_mem
batch_size      = args.batch_size
batch_size_test = args.batch_size*2
num_epochs      = args.num_epochs
num_steps       = args.num_steps
lr   = args.lr


#--------------------------------------------------
# Load dataset
#--------------------------------------------------

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if args.dataset == 'cifar10':
    num_cls = 10
    img_size = 32

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    num_cls = 100
    img_size = 32

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
else:
    print("not implemented yet..")
    exit()

class BinOp():
    def __init__(self, model):
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d = count_Conv2d + 1
        print(count_Conv2d)
        start_range = 1
        end_range = count_Conv2d
        self.bin_range = np.linspace(start_range,
                                     end_range, end_range - start_range + 1) \
            .astype('int').tolist()
        print(self.bin_range)
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        print(self.num_of_params)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                print(m)
                index = index + 1
                print('Making k-bit')
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)
                self.target_modules.append(m.weight)

    def binarization(self):
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True). \
                mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        num_bits = weight_conn
        for index in range(self.num_of_params):
            x = self.target_modules[index].data
            xmax = x.abs().max()
            v0 = 1
            v1 = 2
            v2 = -0.5
            y = num_bits[index]
            x = x.add(v0).div(v1)
            x = x.mul(y).round_()
            x = x.div(y)
            x = x.add(v2)
            x = x.mul(v1)
            n_bits = args.quant
            W_sbits = torch.round(x * 2 ** (n_bits - 1))
            W_sbits = W_sbits / 2 ** (n_bits - 1)
            self.target_modules[index].data = W_sbits

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True) \
                .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True) \
                .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0 - 1.0 / s[1]).mul(n)

def quantize(value, n_bits):
    x = value
    v0 = 1
    v1 = 2
    v2 = -0.5
    y = 2 ** n_bits - 1
    x = x.add(v0).div(v1)
    x = x.mul(y).round_()
    x = x.div(y)
    x = x.add(v2)
    x = x.mul(v1)
    return x

trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)


#--------------------------------------------------
# Instantiate the SNN model and optimizer
#--------------------------------------------------
if args.arch == 'vgg9':
    model = VGG9_Direct(num_steps=num_steps, leak_mem=leak_mem, img_size=32, num_cls=10, input_dim=3)
    model = torch.nn.DataParallel(model)
    model_file = torch.load('../SNN_train_infer_quantization_ela/vgg9_direct_cifar10_t5_epoch100.pth.tar', weights_only=False)
    model.module.load_state_dict(model_file['state_dict'])

elif args.arch == 'vgg11':
    model = SNN_VGG11_TBN(num_steps=num_steps, leak_mem=leak_mem, img_size=img_size, num_cls=num_cls)
else:
    print("not implemented yet..")
    exit()

model = model.cuda()

criterion = nn.CrossEntropyLoss()
best_acc = 0

print('********** SNN training and evaluation **********')
train_loss_list = []
test_acc_list = []

acc_top1, acc_top5 = [], []
model.eval()
n_bits = args.quant
b_size = args.b_size
xbar_size = args.xbar_size
base_r = 19.139e3
neg_wt_bits1 = 3
neg_wt_bits2 = 3

wt1 = quantize(model.module.conv1.weight, n_bits)
wt2 = quantize(model.module.conv2.weight, n_bits)
wt3 = quantize(model.module.conv3.weight, n_bits)
wt4 = quantize(model.module.conv4.weight, n_bits)
wt5 = quantize(model.module.conv5.weight, n_bits)
wt6 = quantize(model.module.conv6.weight, n_bits)
wt7 = quantize(model.module.conv7.weight, n_bits)

neg_w1, A1, W1, W_r = section.create_A_mat_mod(wt1, n_bits, neg_wt_bits1, xbar_size, base_r, b_size)
torch.save([wt1, W1, W_r], './hw_outputs/wts1_q')
print('done saving layer 1')

neg_w2, A2, W, W_r = section.create_A_mat_mod(wt2, n_bits, neg_wt_bits2, xbar_size, base_r, b_size)
torch.save([wt2, W, W_r], './hw_outputs/wts2_q')
print('done saving layer 2')

neg_w3, A3, W, W_r = section.create_A_mat_mod(wt3, n_bits, neg_wt_bits2, xbar_size, base_r, b_size)
torch.save([wt3, W, W_r], './hw_outputs/wts3_q')
print('done saving layer 3')

neg_w4, A4, W, W_r = section.create_A_mat_mod(wt4, n_bits, neg_wt_bits2, xbar_size, base_r, b_size)
torch.save([wt4, W, W_r], './hw_outputs/wts4_q')
print('done saving layer 4')

neg_w5, A5, W, W_r = section.create_A_mat_mod(wt5, n_bits, neg_wt_bits2, xbar_size, base_r, b_size)
torch.save([wt5, W, W_r], './hw_outputs/wts5_q')
print('done saving layer 5')

neg_w6, A6, W, W_r = section.create_A_mat_mod(wt6, n_bits, neg_wt_bits2, xbar_size, base_r, b_size)
torch.save([wt6, W, W_r], './hw_outputs/wts6_q')
print('done saving layer 6')

neg_w7, A7, W, W_r = section.create_A_mat_mod(wt7, n_bits, neg_wt_bits2, xbar_size, base_r, b_size)
torch.save([wt7, W, W_r], './hw_outputs/wts7_q')
print('done saving layer 7')

torch.save([neg_w1, A1, W1], './hw_outputs/L1')
torch.save([neg_w2, A2, W], './hw_outputs/L2')
torch.save([neg_w3, A3, W], './hw_outputs/L3')
torch.save([neg_w4, A4, W], './hw_outputs/L4')
torch.save([neg_w5, A5, W], './hw_outputs/L5')
torch.save([neg_w6, A6, W], './hw_outputs/L6')
torch.save([neg_w7, A7, W], './hw_outputs/L7')

L1 = torch.load('./hw_outputs/L1')
neg_w1, A1, W1 = L1[0], L1[1], L1[2]

L2 = torch.load('./hw_outputs/L2')
neg_w2, A2, W = L2[0], L2[1], L2[2]

L3 = torch.load('./hw_outputs/L3')
neg_w3, A3, W = L3[0], L3[1], L3[2]

L4 = torch.load('./hw_outputs/L4')
neg_w4, A4, W = L4[0], L4[1], L4[2]

L5 = torch.load('./hw_outputs/L5')
neg_w5, A5, W = L5[0], L5[1], L5[2]

L6 = torch.load('./hw_outputs/L6')
neg_w6, A6, W = L6[0], L6[1], L6[2]

L7 = torch.load('./hw_outputs/L7')
neg_w7, A7, W = L7[0], L7[1], L7[2]

neg_W_list = [neg_w1, neg_w2, neg_w3, neg_w4, neg_w5, neg_w6, neg_w7]
A_list = [A1, A2, A3, A4, A5, A6, A7]
neg_bits = [neg_wt_bits1, neg_wt_bits2, neg_wt_bits2, neg_wt_bits2, neg_wt_bits2, neg_wt_bits2, neg_wt_bits2]

print('Weights loaded')
print('Starting Evaluation')
global weight_conn

bits = args.quant

bin_op = BinOp(model)
ADC_precision = args.ADC_precision
weight_conn = np.array([2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1,
                        2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1,
                        2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1])
bin_op.binarization()

with torch.no_grad():
    for j, data in enumerate(testloader, 0):
        print(f'batch {j}')
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        start = time.time()
        out = model(images, j, A_list, neg_W_list, xbar_size, n_bits, neg_bits, b_size, ADC_precision)

        prec1, prec5 = accuracy(out, labels, topk=(1, 5))
        acc_top1.append(float(prec1))
        print(f'time for batch {j} = {time.time() - start}; Accuracy = {np.mean(acc_top1)}')

test_accuracy = np.mean(acc_top1)
print(f"test_accuracy : {test_accuracy}")

sys.exit(0)
