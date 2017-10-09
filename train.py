import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from MyDateSet import MyDataSet
import shutil
import argparse

from resnet import resnet34
from VggModel import vgg11
from  Dense import densenet169
from Inception import Inception3
from squeezenet import squeezenet1_1
from AlexNet import alexnet
from MyNet import Net

from predict import predict

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print("checkpoint save" + str(is_best))
    if is_best:
        print("save the best model")
        shutil.copyfile(filename, filename+'model_best.pth.tar')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="mynet",
                        help='Path to model to use')
    parser.add_argument('--cuda', type=int, default=0,
                        help='use cuda ? default is 0')
    args = parser.parse_args()

    net = None
    if(args.model == 'agg11'):
        net = vgg11(pretrained=False)
    elif(args.model == 'resnet34'):
        net = resnet34()
    elif(args.model == 'squeezenet'):
        net = squeezenet1_1()
    elif(args.model == 'dense'):
        net = densenet169()
    elif(args.model == 'alexnet'):
        net = alexnet()
    elif(args.model == 'inception3'):  # this model can not be used now, some problem to be done
        net = Inception3()
    else:
        net = Net()
        args.model = 'mynet'

    print('-' * 10 + 'use the model : ' + args.model + '-' * 10)
    'now load the data'
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Scale(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                                     std=[0.229, 0.224, 0.225])])
    train_data_set = MyDataSet(transform = transform)
    train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=8, shuffle=True, num_workers=4)
    'dataloader init over'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)# momentum是在梯度前加一个系数

    best_prec1 = 100
    if args.cuda > -1 and torch.cuda.is_available():
        net.cuda(args.cuda)
        criterion.cuda()

    'start train'
    for epoch in range(30):
        running_loss = 0.0
        for i ,data in enumerate(train_data_loader, 0):
            inputs ,labels = data

            inputs ,labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            if args.cuda > -1 and torch.cuda.is_available():
                inputs.cuda()
                labels.cuda()

            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99: # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 100))
                prec1 = running_loss / 100

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'best_prec1': best_prec1,
                }, prec1 < best_prec1, filename=args.model+"checkpoint.pth.tar")
                best_prec1 = min(prec1, best_prec1)
                running_loss = 0.0

    'train over'
    torch.save(net.state_dict(), args.model + 'model')
    'now predicting'
    predict(net, args.model, args.cuda)
    'predicting over'
