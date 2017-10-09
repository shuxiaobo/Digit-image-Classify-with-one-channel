import numpy as np
import torch.utils.data
from torch.autograd import Variable
from MyDateSet import MyDataSet
import torchvision.transforms as transforms
import csv
import argparse

from resnet import resnet34
from VggModel import vgg11
from  Dense import densenet169
from Inception import Inception3
from squeezenet import squeezenet1_1
from AlexNet import alexnet
from MyNet import Net
def predict(net, modelname, cuda):

    # ok ,now let's get the predicted answer
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Scale(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                                     std=[0.229, 0.224, 0.225])])
    test_data_set = MyDataSet(train=False, transform = transform)
    test_data_loader = torch.utils.data.DataLoader(test_data_set, batch_size=4, shuffle=False, num_workers=4)
    result = []

    for data in test_data_loader:
        x ,labels = data
        x = Variable(x)
        if cuda > -1 and torch.cuda.is_available():
            x = x.cuda()
        output = net(x)
        _, predicted = torch.max(output.data, 1)
        result.extend(predicted.tolist())
    results = dict((i + 1, result[i]) for i in range(len(result)))
    with open(modelname+'entry.csv', 'w', newline='') as csvfile:
        csv_w = csv.writer(csvfile)
        csv_w.writerow(('Id', 'Label'))
        for row in sorted(results.items()):
            csv_w.writerow(row)

    print(result)

if __name__ == "__main__":
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
    if args.cuda > -1 and torch.cuda.is_available():
        net.cuda()
    net.load_state_dict(torch.load(args.model+'checkpoint.pth.tar')['state_dict'])
    predict(net, args.model, args.cuda)