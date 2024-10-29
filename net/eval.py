import os
import sys
import torch
import cv2
import numpy as np
import torch.nn as nn
import torchvision.utils as vutils
from utils import decode_segmap
from SegDataFolder import semData
from getSetting import get_yaml, get_criterion, get_optim, get_scheduler, get_net
from metric import AverageMeter, intersectionAndUnion
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve


# for transform
# 需要计算文件夹对应图片的mean&std，或者使用trainset的mean std
# mean = np.array([0.03369877519145767, 0.02724876914897716, 0.019425700166401196, 0.03361996144023702,
#                       0.032213421442935264, 0.03168874880981447, 0.031527547610634275])       # 训练原图的平均值
# std = np.array([0.0033888922358425873, 0.0038459901628366544, 0.004287149017973415, 0.01173405280221371,
#                       0.3572610932095403, 0.3572878634587769, 0.3572964354695759])      # 训练原图的方差"""
mean = np.array([0.019051681568420666,-2.8460242480404057e-05,-5.739556561535955e-05,0.0017318990755127608,36.43442654079862,169.33292134602863,125.13270331488711])

std = np.array([0.05090688233052699,0.004969403668858973,0.004973975379152158,0.002214399183101169,17.74314659760822,63.963608893449425,56.74846886453103])
def get_idx(channels):
    assert channels in [2, 4, 7, 8, 3]
    if channels == 3:
        return list(range(3))
    elif channels == 4:
        return list(range(4))
    elif channels == 7:
        return list(range(7))
    elif channels == 8:
        return list(range(8))
    elif channels == 2:
        return list(range(6))[-2:]

def getTransorm4eval(channel_idx=[0,1,2,3]):
    import transform as T
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=mean[channel_idx],std=std[channel_idx])
        ]
    )
def convert_to_color(image, output_path):
    # 创建一个空的彩色图像
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # 将三个值分别转换为不同的颜色
    # 假设值是0, 1, 2
    # 0 -> 蓝色 (255, 0, 0)
    # 1 -> 绿色 (0, 255, 0)
    # 2 -> 红色 (0, 0, 255)
    color_image[np.where(image == 0)] = [0, 0, 0]
    color_image[np.where(image == 1)] = [0, 255, 0]
    color_image[np.where(image == 2)] = [0, 0, 255]

    # 保存彩色图像
    cv2.imwrite(output_path, color_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def eval(args,config,):
    os.makedirs(args.outputdir, exist_ok=True)

    net = get_net(config) 
    softmax = nn.Softmax(dim=1)
    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['net'])
    net = net.cuda() if args.use_gpu else net

    dataset = semData(
        train=False,
        root='./Data',
        channels = config['channels'],
        transform=getTransorm4eval(channel_idx=get_idx(config['channels'])),
        selftest_dir=args.testdir
    )
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False)
    
    net.eval()
    with torch.no_grad():
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        correct = []
        preds = []



        for i, batch in enumerate(dataloader):
            img = batch['X'].cuda() if args.use_gpu else batch['X']
            label = batch['Y'].cuda() if args.use_gpu else batch['Y']
            path = batch['path'][0].split('.')[0]
            # label[label == 3] = 0

            outp = net(img)
            
            score = softmax(outp)      
            pred = score.max(1)[1]
            # pred[pred == 3] = 0

            saved_1 = pred.squeeze().cpu().numpy()
            saved_255 = 122 * saved_1

            # 保存预测图片，像素值为0&1
            cv2.imwrite(r'E:\nxy\net\eval_output\eval_output_1/{}.png'.format(path), saved_1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # 保存预测图片，像素值为0&255
            cv2.imwrite(r'E:\nxy\net\eval_output\eval_output_255/{}.png'.format(path), saved_255, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            # 将三值图像转换为彩色图像并保存
           # convert_to_color(saved_1, r'D:\BARELAND\NET\eval_output\eval_ouput_color/{}.tif'.format(path))
            
            correct.extend(label.reshape(-1).cpu().numpy())
            preds.extend(pred.reshape(-1).cpu().numpy())
            
            N = img.size()[0]
            
            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), label.cpu().numpy(), config['n_classes'], config['ignore_index'])
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            acc = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

        
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        acc_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(acc_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum)+1e-10)
        precision = precision_score(correct, preds, average='binary')
        recall = recall_score(correct, preds, average='binary')
        f1 = f1_score(correct, preds, average='binary')
        f1_scores_per_class = f1_score(correct, preds, average=None)
        for i, score in enumerate(f1_scores_per_class):
            print(f"Class {i} F1 Score: {score}")

        print(iou_class)
        print(acc_class)
        print('mIoU {:.4f} | mAcc {:.4f} | allAcc {:.4f} | precision {:.4f} | recall {:.4f} | f1 {:.4f}'.format(
            mIoU, mAcc, allAcc, precision, recall, f1))
        # print("小麦总产量：2688449斤 、油菜总产量：150391斤")
        

### argument parse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--testdir',type=str,default='eval', help='path of test images') # 测试文件夹，需要放在Data文件夹下，跟train,test有相同结构
parser.add_argument('--outputdir',type=str,default='eval_output',help='test output') # 指定生成预测图片的输出文件夹
parser.add_argument('--gpu',type=int, default=0,help='gpu index to run') # gpu卡号，单卡默认为0
parser.add_argument('--configfile',type=str, default='ConfigFiles/config-deeplab.yaml',help='path to config files') # 选定configfile,用于指定网络
parser.add_argument('--checkpoint',type=str, default=r"E:\nxy\net\deeplab+7_2Adam_lr=0.0001\DeepLab-0.8160-ep99.pth", help='checkpoint path') # 加载定模型路径
args = parser.parse_args()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if torch.cuda.is_available():
        args.use_gpu = True
    else:
        args.use_gpu = False
    
    config = get_yaml(args.configfile)
    eval(args, config)


