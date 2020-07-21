import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import matplotlib.pyplot as plt
import numpy as np
import models
# from models import sync_bn
import dataset as ds
from options.options import parser
import torch.nn.functional as F

best_mIoU = 0


def main():
    global args, best_mIoU
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)

    if args.dataset == 'VOCAug' or args.dataset == 'VOC2012' or args.dataset == 'COCO':
        num_class = 21
        ignore_label = 255
        scale_series = [10, 20, 30, 60]
    elif args.dataset == 'Cityscapes':
        num_class = 19
        ignore_label = 255 
        scale_series = [15, 30, 45, 90]
    elif args.dataset == 'ApolloScape':
        num_class = 37 
        ignore_label = 255 
    elif args.dataset == 'CULane':
        num_class = 5
        ignore_label = 4
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = models.ERFNet(num_class)
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code

    # test_loader = torch.utils.data.DataLoader(
    #     getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(data_list=args.val_list, transform=torchvision.transforms.Compose([
    #         tf.GroupRandomScaleNew(size=(args.img_width, args.img_height), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
    #         tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
    #     ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    test_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(data_list=args.val_list, transform=torchvision.transforms.Compose([
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) optimizer and evaluator
    weights = [0.24, 0.24, 0.24, 0.04, 0.24]
    # weights = [1.0 for _ in range(5)]
    # weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)

    ### evaluate ###
    validate(test_loader, model, criterion, 0, evaluator)
    return


def validate(val_loader, model, criterion, iter, evaluator, logger=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

      input_var = torch.autograd.Variable(input, volatile=True)

      # compute output
      # output, output_exist = model(input_var)
      output = model(input_var)

      # measure accuracy and record loss

      output = F.log_softmax(output, dim=1)

      pred = output.data.cpu().numpy() # BxCxHxW
      pred_loss = output.data.cpu().numpy().transpose(0, 2, 3, 1)
      pred_loss = np.argmax(pred_loss, axis=3).astype(np.uint8)
      IoU.update(evaluator(pred_loss, target.cpu().numpy()))

      for idx, pred_int in enumerate(pred):
        pred_int = pred_int.argmax(axis=0)
        pred_rgb = label_to_rgb(pred_int, color_encoding)
        directory = args.outdir
        if not os.path.exists(directory):
          os.makedirs(directory)
        output = os.path.join(directory, "batch_" + str(i) + "_" + str(idx) + ".png")
        plt.imsave(output,pred_rgb)


      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
      if (i + 1) % args.print_freq == 0:
        print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time)))

    acc = np.sum(np.diag(IoU.sum)) / float(np.sum(IoU.sum))
    mIoU = np.diag(IoU.sum) / (1e-20 + IoU.sum.sum(1) + IoU.sum.sum(0) - np.diag(IoU.sum))
    mIoU = np.sum(mIoU) / len(mIoU)
    print(('Testing Results: Pixels Acc {acc:.3f}\tmIoU {mIoU:.3f} ({bestmIoU:.4f})'.format(acc=acc, mIoU=mIoU, bestmIoU=max(mIoU, best_mIoU))))
      

    return mIoU


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

class_name = {0: 'human',
      1: 'vehicle',
      2: 'movable_object',
      3: 'background',
      4: 'other'}


color_encoding = {0: (255, 0, 0),
      1: (255, 255, 0),
      2: (0, 0, 255),
      3: (0, 255, 0),
      4: (255, 0, 255)}

def rgb_to_label(rgb_image, colormap = color_encoding):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]
    encoded_image = np.zeros(shape,dtype=np.int8)
    for i, cls in enumerate(colormap):
        for x in range(encoded_image.shape[0]):
            for y in range (encoded_image.shape[1]):
                if(np.all(rgb_image[x][y] == colormap[i])):
                    encoded_image[x][y] = i
    return encoded_image


def label_to_rgb(label, colormap = color_encoding):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    output = np.zeros(label.shape[:2]+(3,))
    for k in colormap.keys():
        output[label==k] = colormap[k]
    return np.uint8(output)

if __name__ == '__main__':
    main()
