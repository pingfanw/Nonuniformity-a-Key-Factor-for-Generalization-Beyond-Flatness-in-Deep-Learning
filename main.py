import argparse
import os
import csv

import torch
import torch.nn as nn

from tqdm import tqdm
from timm.loss import LabelSmoothingCrossEntropy
from utils.network_utils_cifar import get_network_cifar
from utils.network_utils_mnist import get_network_mnist
from utils.network_utils_imagenet import get_network_imagenet
from utils.data_utils import NoiseDataLoader
from optimizer import sgd_mod
from optimizer.minimizer import GAM, SAM, ASAM, CSAM, CSAM_Identity, ACSAM_Identity

def prepare_csv(trainlog_path, testlog_path, minimizer, rho, noise_rate):
    if not os.path.exists(trainlog_path):
        os.makedirs(trainlog_path)
    if not os.path.exists(testlog_path):
        os.makedirs(testlog_path)
    if noise_rate==0.0:
        csv_train = open(trainlog_path+'/'+minimizer.lower()+str(rho)+'.csv', 'a+', newline='')
        csv_test = open(testlog_path+'/'+minimizer.lower()+str(rho)+'.csv', 'a+', newline='')
    else:
        csv_train = open(trainlog_path+'/'+minimizer.lower()+str(rho)+'_noise'+str(noise_rate)+'.csv', 'a+', newline='')
        csv_test = open(testlog_path+'/'+minimizer.lower()+str(rho)+'_noise'+str(noise_rate)+'.csv', 'a+', newline='')
    csv_train_writer = csv.writer(csv_train)
    csv_test_writer = csv.writer(csv_test)
    return csv_train, csv_train_writer, csv_test, csv_test_writer

# ----------train----------#
def train(args):
    noisedataloader = NoiseDataLoader(args.dataset.lower(), 
                                      args.noise_rate, 
                                      args.noise_mode, 
                                      args.train_batch_size, 
                                      args.test_batch_size, 
                                      args.num_workers, 
                                      args.data_path)
    trainloader, testloader = noisedataloader.get_loader()

    nc = { 
        'mnist': 10,
        'fashionmnist': 10,
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000
    }
    # ns_train = len(trainloader.dataset)
    args.outputs_dim = nc[args.dataset.lower()]
    if args.dataset.lower() == 'mnist' or args.dataset.lower() == 'fashionmnist':
        net = get_network_mnist(args.network)
    if args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar100':
        if args.network == 'preresnet':
            net = get_network_cifar(args.network,
                          depth=args.depth,
                          num_classes=args.outputs_dim)
        elif args.network == 'pyramidnet':
            net = get_network_cifar(args.network,
                          depth = args.depth,
                          alpha = 48,
                          input_shape = (1, 3, 32, 32),
                          num_classes = args.outputs_dim,
                          base_channels = 16,
                          block_type = 'bottleneck')
        else:
            net = get_network_cifar(args.network,
                          depth=args.depth,
                          num_classes=args.outputs_dim,
                          growthRate=args.growthRate,
                          compressionRate=args.compressionRate,
                          widen_factor=args.widen_factor,
                          dropRate=args.dropRate)
    if args.dataset.lower() == 'imagenet':
        if args.network == 'pyramidnet':
            net = get_network_imagenet(args.network,
                            depth=args.depth,
                            alpha=48,
                            input_shape=(1, 3, 224, 224),
                            num_classes=args.outputs_dim,
                            base_channels=16,
                            block_type='bottleneck')
        else:
            net = get_network_imagenet(args.network,
                          depth=args.depth,
                          num_classes=args.outputs_dim,
                          growthRate=args.growthRate,
                          compressionRate=args.compressionRate,
                          widen_factor=args.widen_factor,
                          dropRate=args.dropRate)
    net = net.to(args.device)

    if args.smoothing:
        loss_function = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        loss_function = nn.CrossEntropyLoss()
    if args.network.lower()=='preresnet':
        optimizer = sgd_mod.SGD_nonfull(net, args.minimizer, 
                                        lr=args.learning_rate, 
                                        weight_decay=args.weight_decay, 
                                        momentum=args.momentum, 
                                        stat_decay=args.stat_decay, 
                                        damping=args.damping, 
                                        batch_averaged=args.batch_averaged, 
                                        TCov=args.TCov, TInv=args.TInv)
    else:
        optimizer = sgd_mod.SGD(net, args.minimizer, 
                                lr=args.learning_rate, 
                                weight_decay=args.weight_decay, 
                                momentum=args.momentum, 
                                stat_decay=args.stat_decay, 
                                damping=args.damping, 
                                batch_averaged=args.batch_averaged, 
                                TCov=args.TCov, TInv=args.TInv)
    if args.minimizer == 'CSAM' or args.minimizer == 'CSAM_Identity' or args.minimizer == 'ACSAM_Identity':
        minimizer = eval(args.minimizer)(optimizer, net, first_rho=args.first_rho, second_rho=args.second_rho, consistent_momentum=args.consistent_momentum)
    else:
        minimizer = eval(args.minimizer)(optimizer, net, rho=args.rho, eta=args.eta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(minimizer.optimizer, args.epoch_range)

    csv_train, csv_train_writer, csv_test, csv_test_writer = prepare_csv(args.trainlog_path, args.testlog_path, args.minimizer, args.rho, args.noise_rate)
    csv_train_writer.writerow([
        'network:',args.network,'depth',args.depth,'Loss:CrossEntropy','DataSet:',args.dataset,'LearningRate:',
        args.learning_rate,'BatchSize:',args.train_batch_size,'EpochRange:',args.epoch_range,'Optimizer:SGD','Minimizer:',args.minimizer])
    csv_train_writer.writerow(['Epoch','Train_Loss', 'Disturb_Loss', 'Train_Accuracy'])
    csv_train.flush()
    csv_test_writer.writerow([
        'network:',args.network,'depth',args.depth,'Loss:CrossEntropy','DataSet:',args.dataset,'Optimizer:SGD','Minimizer:',args.minimizer])
    csv_test_writer.writerow(['Epoch', 'Test_Loss','Test_Accuracy','Generalization_Gap'])
    csv_test.flush()

    # train
    optimizer.acc_stats = True
    for epoch in range(args.epoch_range):
        total = 0
        num_correct = 0
        train_loss = 0.0
        disturb_loss = 0.0
        net.train()
        desc = ('[Train][%s][%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' % 
                (args.minimizer.lower(), epoch+1, scheduler.get_last_lr(), 0, 0, num_correct, total))
        prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, position=0, leave=True)
        for batch_index, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            if args.network == 'mlp': inputs = inputs.view(-1, 784)
            outputs = net.forward(inputs)
            predictions = net(inputs)
            if args.minimizer == "GAM":
                # Ascent Step
                loss = loss_function(net(inputs), labels)
                optimizer.acc_stats = True
                optimizer.zero_grad()
                loss.backward()
                minimizer.ascent_step()
                # Descent Step
                loss_eps = loss_function(net(inputs), labels)  
                loss_eps.backward()   
                optimizer.acc_stats = False
                minimizer.descent_step()    
            elif args.minimizer == "CSAM":
                # first ascent step
                loss_first = loss_function(net(inputs), labels)
                loss_first.backward()
                original_grad = minimizer.first_ascent_step()
                # second ascent step
                loss = loss_function(net(inputs), labels)
                optimizer.acc_stats = True
                optimizer.zero_grad()
                loss.backward()
                minimizer.second_ascent_step(original_grad)
                # Descent Step
                loss_eps = loss_function(net(inputs), labels)
                loss_eps.backward()
                optimizer.acc_stats = False
                minimizer.descent_step()
            elif args.minimizer == "CSAM_Identity" or args.minimizer == "ACSAM_Identity":
                # first ascent step
                loss_first = loss_function(net(inputs), labels)
                loss_first.backward()
                original_grad = minimizer.first_ascent_step()
                # second ascent step
                loss = loss_function(net(inputs), labels)
                loss.backward()
                minimizer.second_ascent_step(original_grad)
                # Descent Step
                loss_eps = loss_function(net(inputs), labels)
                loss_eps.backward()
                minimizer.descent_step()
            else:
                # Ascent Step
                loss = loss_function(outputs, labels)
                loss.backward()
                minimizer.ascent_step()
                # Descent Step
                loss_eps = loss_function(net(inputs), labels)
                loss_eps.backward()
                minimizer.descent_step()
            # visualizing and saving stuff
            predictions = outputs.argmax(dim=1) 
            total += labels.size(0)
            num_correct += torch.eq(predictions,labels).sum().item()  
            train_loss += (loss.item())
            disturb_loss += (loss_eps.item())
            desc = ('[Train][%s][%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (args.minimizer.lower(), 
                     epoch+1,
                     scheduler.get_last_lr(), 
                     train_loss / (batch_index + 1), 
                     100. * num_correct / total, 
                     num_correct, total))
            prog_bar.set_description(desc, refresh=True)
            prog_bar.update()
        prog_bar.close()
        csv_train_writer.writerow([epoch+1, 
                                   train_loss / (batch_index + 1), 
                                   disturb_loss / (batch_index + 1), 
                                   100.*num_correct / len(trainloader.dataset)])
        csv_train.flush()
        scheduler.step()

        # test  
        net.eval()
        total = 0
        num_correct = 0
        test_loss = 0.0
        with torch.no_grad():
            desc = ('[Test][%s][LR=%s] Loss: %.3f | Gap: %.3f | Acc: %.3f%% (%d/%d)' % 
                    (args.minimizer.lower(), scheduler.get_last_lr(), test_loss/(0+1), 0, 0, num_correct, total))
            prog_bar_test = tqdm(enumerate(testloader), total=len(testloader), desc=desc, position=0, leave=True)
            for batch_index, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                if args.network == 'mlp': inputs = inputs.view(-1, 784)
                outputs = net.forward(inputs)
                predictions = outputs.argmax(dim=1)
                total += labels.size(0)
                num_correct += torch.eq(predictions, labels).sum().item()
                test_loss += loss_function(outputs, labels).item()
                desc = ('[Test][%s][LR=%s] Loss: %.3f | Gap: %.3f | Acc: %.3f%% (%d/%d)' %
                        (args.minimizer.lower(), 
                         scheduler.get_last_lr(), 
                         test_loss / (batch_index + 1), 
                         abs((test_loss / (batch_index+1)) - (train_loss)), 
                         100. * num_correct / total, 
                         num_correct, total))
                prog_bar_test.set_description(desc, refresh=True)
                prog_bar_test.update()
        prog_bar_test.close()
        csv_test_writer.writerow([epoch+1,
                                  test_loss / (batch_index+1), 
                                  100.*num_correct / len(testloader.dataset), 
                                  abs((test_loss / (batch_index+1)) - (train_loss))
        ])
        csv_test.flush()
    csv_train.close()
    csv_test.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIFAR10',type=str, help=" CIFAR10 CIFAR100 MNIST FASHIONMNIST IMAGNET.")
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--noise_rate', default=0.0, type=float, help="Noise Rate.")
    parser.add_argument('--noise_mode', default='sym', type=str, help="Noise Mode for sym and asym.")
    parser.add_argument('--num_workers', default=0, type=int, help="Num of workers.")
    parser.add_argument('--data_path', default='./data', type=str, help="Data Path of Standard Datasets. Imagenet: D:/Projects/Python/Datasets/ImageNet")
    parser.add_argument('--outputs_dim', default=10, type=int, help="Dimension of outputs.")
    parser.add_argument('--depth', default=16, type=int, help="Depth of net.")
    parser.add_argument('--network', default='vgg16_bn', type =str)
    parser.add_argument('--train_batch_size', default=128, type=int, help="Train Batch size")
    parser.add_argument('--test_batch_size', default=128, type=int, help="Test Batch size")
    parser.add_argument('--epoch_range', default=300, type=int, help="Num of Epoch")
    parser.add_argument('--device', default='cuda', type=str, help="CUDA or CPU")  
    parser.add_argument('--trainlog_path',default='')
    parser.add_argument('--testlog_path',default='')  
    # wrn, densenet
    parser.add_argument('--widen_factor', default=1, type=int)      
    parser.add_argument('--dropRate', default=0.1, type=float)
    parser.add_argument('--growthRate', default=2, type=int)       
    parser.add_argument('--compressionRate', default=2, type=int)
    # pyramid
    parser.add_argument('--alpha', default=48, type=int)
    parser.add_argument('--block_type', default='basic', type=str)
    parser.add_argument('--base_channels', default=16, type=int)
    # other argument
    parser.add_argument('--resume', '-r', action='store_true')         # resume from checkpoint
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    # minimizer argument
    parser.add_argument('--minimizer', default='GAM', type=str, help="GAM/CSAM/CSAM_Identity/ASAM/SAM.")
    parser.add_argument('--rho', default=0.01, type=float, help="Rho for SAM/GAM/ASAM.")
    parser.add_argument('--eta', default=0.0, type=float, help="Eta for ASAM.")
    parser.add_argument('--first_rho', default=0.1, type=float, help="Rho for CSAM and CSAM_Identity.")
    parser.add_argument('--second_rho', default=0.1, type=float, help="Rho for CSAM and CSAM_Identity.")
    parser.add_argument('--consistent_momentum', default=0.9, type=float, help="Consistent Momentum for CSAM and CSAM_Identity.")
    # optimizer argument
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float, help="Weight decay factor.")
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--stat_decay', default=0.95, type=float)         
    parser.add_argument('--damping', default=1e-3, type=float)
    parser.add_argument('--batch_averaged', default=True, type=bool)       
    parser.add_argument('--TCov', default=50, type=int)
    parser.add_argument('--TInv', default=50, type=int)



    """ MNIST """
    dataset = 'MNIST'
    # mlp
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'mlp'
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.9
        rho = [1e-1, 1.0, 1e-1, 2e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+'/test' 
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--dataset', dataset,
            '--smoothing', 0.0,
            '--epoch_range', str(300),
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)


    """ CIFAR10 """    
    dataset = 'CIFAR10'
    # wrn58-6
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'wrn'
        depth = 58
        widen_factor = 6
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.6
        rho = [1e-1, 1.0, 1e-1, 2e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+str(widen_factor)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+str(widen_factor)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--widen_factor', str(widen_factor),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
    # wr28-6
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'wrn'
        depth = 28
        widen_factor = 6
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.9
        rho = [1e-1, 1.0, 1e-1, 2e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+str(widen_factor)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+str(widen_factor)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--widen_factor', str(widen_factor),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
    # rn-20
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'resnet'
        depth = 20
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.6
        rho = [1e-1, 1.0, 1e-1, 2e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
    # rn-56
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'resnet'
        depth = 56
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.6
        rho = [1e-1, 1.0, 1e-1, 2e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
    # pyramidnet-110
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'pyramidnet'
        depth = 110
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.4
        rho = [1e-1, 1.0, 1e-1, 2e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--alpha', str(16),
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
    # densenet-100
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'densenet'
        depth = 100
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.9
        rho = [1e-1, 1.0, 1e-1, 2e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)


    """ CIFAR100 """    
    dataset = 'CIFAR100'
    # wr28-6 
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'wrn'
        depth = 28
        widen_factor = 6
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.9
        rho = [1e-1, 1.0, 1e-1, 4e-1]
        train_bs = [128,160,160]
        test_bs = [128,160,160]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+str(widen_factor)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+str(widen_factor)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--widen_factor', str(widen_factor),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
    # wrn58-6
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'wrn'
        depth = 58
        widen_factor = 6
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.6
        rho = [1e-1, 1.0, 1e-1, 2e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+str(widen_factor)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+str(widen_factor)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--widen_factor', str(widen_factor),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
    # rn-20
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'resnet'
        depth = 20
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.2
        rho = [1e-1, 1.0, 1e-1, 4e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
    # rn-56
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'resnet'
        depth = 56
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.9
        rho = [1e-1, 1.0, 1e-1, 4e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
    # pyramidnet-110
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'pyramidnet'
        depth = 110
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.5
        rho = [1e-1, 1.0, 1e-1, 2e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--alpha', str(16),
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
    densenet-100 reruning
    for i in range(3):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'densenet'
        depth = 100
        minimizer = ['CSAM_Identity','ASAM', 'SAM']
        consistent_momentum = 0.9
        rho = [1e-1, 1.0, 1e-1, 4e-1]
        train_bs = [128,128,128]
        test_bs = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[3]),
            '--consistent_momentum', str(consistent_momentum),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)


    """ CIFAR10-Noise """
    dataset = 'CIFAR10'
    # wrn28-6 CSAM and SAM
    noise_rate = [0.2,0.4,0.6,0.8]
    for h in range(len(noise_rate)):
        for i in range(2):
            learning_rate = [1e-1,1e-1]
            network = 'wrn'
            depth = 28
            widen_factor = 6
            minimizer = ['CSAM_Identity','SAM']
            rho = [1e-1, 1e-1, 2e-1]
            consistent_momentum = 0.9
            train_bs = [128,128]
            test_bs = [128,128]
            trainlog_path = 'logs/'+dataset.lower()+'noise/'+network.lower()+str(depth)+str(widen_factor)+'/train'
            testlog_path = 'logs/'+dataset.lower()+'noise/'+network.lower()+str(depth)+str(widen_factor)+'/test'
            args_train = parser.parse_known_args(args=[
                '--learning_rate', str(learning_rate[i]),
                '--network', network,
                '--depth', str(depth),
                '--widen_factor', str(widen_factor),
                '--dataset', dataset,
                '--noise_rate', str(noise_rate[h]),
                '--noise_mode', 'sym',
                '--minimizer', minimizer[i],
                '--rho', str(rho[i]),
                '--first_rho', str(rho[i]),
                '--second_rho', str(rho[2]),
                '--consistent_momentum', str(consistent_momentum),
                '--train_batch_size',str(train_bs[i]),
                '--test_batch_size',str(test_bs[i]),
                '--trainlog_path', trainlog_path,
                '--testlog_path', testlog_path])[0]
            assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
                f"Invalid minimizer type. Please select the correct minimizer"
            train(args_train)
    # wrn28-6 ACSAM and SAM
    noise_rate = [0.2,0.4,0.6,0.8]
    for h in range(len(noise_rate)):
        for i in range(1):
            learning_rate = [1e-1,1e-1]
            network = 'wrn'
            depth = 28
            widen_factor = 6
            minimizer = ['ACSAM_Identity','SAM']
            rho = [1e-1, 1e-1, 2e-1]
            consistent_momentum = 0.9
            train_bs = [128,128]
            test_bs = [128,128]
            trainlog_path = 'logs/'+dataset.lower()+'noise/'+network.lower()+str(depth)+str(widen_factor)+'/train'
            testlog_path = 'logs/'+dataset.lower()+'noise/'+network.lower()+str(depth)+str(widen_factor)+'/test'
            args_train = parser.parse_known_args(args=[
                '--learning_rate', str(learning_rate[i]),
                '--network', network,
                '--depth', str(depth),
                '--widen_factor', str(widen_factor),
                '--dataset', dataset,
                '--noise_rate', str(noise_rate[h]),
                '--noise_mode', 'sym',
                '--minimizer', minimizer[i],
                '--rho', str(rho[i]),
                '--first_rho', str(rho[i]),
                '--second_rho', str(rho[2]),
                '--consistent_momentum', str(consistent_momentum),
                '--train_batch_size',str(train_bs[i]),
                '--test_batch_size',str(test_bs[i]),
                '--trainlog_path', trainlog_path,
                '--testlog_path', testlog_path])[0]
            assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
                f"Invalid minimizer type. Please select the correct minimizer"
            train(args_train)


    """ CIFAR10-rho """
    dataset = 'CIFAR10'
    # wrn28-6 CSAM and SAM  
    rho = [0.05,0.1,0.15,0.2,0.25]
    for i in range(len(rho)):
        minimizer = ['CSAM_Identity','SAM']
        first_rho = 1e-1
        consistent_momentum = 0.9
        for j in range(len(minimizer)):
            learning_rate = 0.1
            network = 'wrn'
            depth = 28
            widen_factor = 6
            train_bs = [128,128]
            test_bs = [128,128]
            trainlog_path = 'logs/'+dataset.lower()+'rho/'+network.lower()+str(depth)+str(widen_factor)+'/train'
            testlog_path = 'logs/'+dataset.lower()+'rho/'+network.lower()+str(depth)+str(widen_factor)+'/test'
            args_train = parser.parse_known_args(args=[
                '--learning_rate', str(learning_rate),
                '--network', network,
                '--depth', str(depth),
                '--widen_factor', str(widen_factor),
                '--dataset', dataset,
                '--minimizer', minimizer[j],
                '--rho', str(rho[i]),
                '--first_rho', str(first_rho),
                '--second_rho', str(rho[i]),
                '--consistent_momentum', str(consistent_momentum),
                '--train_batch_size',str(train_bs[j]),
                '--test_batch_size',str(test_bs[j]),
                '--trainlog_path', trainlog_path,
                '--testlog_path', testlog_path])[0]
            assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
                f"Invalid minimizer type. Please select the correct minimizer"
            train(args_train)
    # pyramidnet-110 CSAM and SAM
    rho = [0.05,0.1,0.15,0.2,0.25]
    for i in range(len(rho)):
        minimizer = ['CSAM_Identity','SAM']
        first_rho = 1e-1
        consistent_momentum = 0.9
        for j in range(len(minimizer)):
            learning_rate = 0.1
            network = 'pyramidnet'
            depth = 110
            train_bs = [128,128]
            test_bs = [128,128]
            trainlog_path = 'logs/'+dataset.lower()+'rho/'+network.lower()+str(depth)+'/train'
            testlog_path = 'logs/'+dataset.lower()+'rho/'+network.lower()+str(depth)+'/test'
            args_train = parser.parse_known_args(args=[
                '--learning_rate', str(learning_rate),
                '--network', network,
                '--depth', str(depth),
                '--dataset', dataset,
                '--minimizer', minimizer[j],
                '--alpha', str(16),
                '--rho', str(rho[i]),
                '--first_rho', str(first_rho),
                '--second_rho', str(rho[i]),
                '--consistent_momentum', str(consistent_momentum),
                '--train_batch_size',str(train_bs[j]),
                '--test_batch_size',str(test_bs[j]),
                '--trainlog_path', trainlog_path,
                '--testlog_path', testlog_path])[0]
            assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
                f"Invalid minimizer type. Please select the correct minimizer"
            train(args_train)

    """ CIFAR100-rho """
    dataset = 'CIFAR100'
    # wrn28-6 CSAM and SAM
    rho = [0.1,0.2,0.3,0.4,0.5]
    for i in range(len(rho)):
        minimizer = ['CSAM_Identity','SAM']
        first_rho = 1e-1
        consistent_momentum = 0.9
        for j in range(len(minimizer)):
            learning_rate = 0.1
            network = 'wrn'
            depth = 28
            widen_factor = 6
            train_bs = [128,128]
            test_bs = [128,128]
            trainlog_path = 'logs/'+dataset.lower()+'rho/'+network.lower()+str(depth)+str(widen_factor)+'/train'
            testlog_path = 'logs/'+dataset.lower()+'rho/'+network.lower()+str(depth)+str(widen_factor)+'/test'
            args_train = parser.parse_known_args(args=[
                '--learning_rate', str(learning_rate),
                '--network', network,
                '--depth', str(depth),
                '--widen_factor', str(widen_factor),
                '--dataset', dataset,
                '--minimizer', minimizer[j],
                '--rho', str(rho[i]),
                '--first_rho', str(first_rho),
                '--second_rho', str(rho[i]),
                '--consistent_momentum', str(consistent_momentum),
                '--train_batch_size',str(train_bs[j]),
                '--test_batch_size',str(test_bs[j]),
                '--trainlog_path', trainlog_path,
                '--testlog_path', testlog_path])[0]
            assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
                f"Invalid minimizer type. Please select the correct minimizer"
            train(args_train)
    # pyramidnet-110 CSAM and SAM
    rho = [0.1,0.2,0.3,0.4,0.5]
    for i in range(len(rho)):
        minimizer = ['CSAM_Identity','SAM']
        first_rho = 1e-1
        consistent_momentum = 0.9
        for j in range(len(minimizer)):
            learning_rate = 0.1
            network = 'pyramidnet'
            depth = 110
            train_bs = [128,128]
            test_bs = [128,128]
            trainlog_path = 'logs/'+dataset.lower()+'rho/'+network.lower()+str(depth)+'/train'
            testlog_path = 'logs/'+dataset.lower()+'rho/'+network.lower()+str(depth)+'/test'
            args_train = parser.parse_known_args(args=[
                '--learning_rate', str(learning_rate),
                '--network', network,
                '--depth', str(depth),
                '--dataset', dataset,
                '--minimizer', minimizer[j],
                '--alpha', str(16),
                '--rho', str(rho[i]),
                '--first_rho', str(first_rho),
                '--second_rho', str(rho[i]),
                '--consistent_momentum', str(consistent_momentum),
                '--train_batch_size',str(train_bs[j]),
                '--test_batch_size',str(test_bs[j]),
                '--trainlog_path', trainlog_path,
                '--testlog_path', testlog_path])[0]
            assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
                f"Invalid minimizer type. Please select the correct minimizer"
            train(args_train)

    """ Imagenet """
    dataset = 'IMAGENET'
    # rn-20
    for i in range(2):
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'resnet'
        depth = 20
        minimizer = ['CSAM_Identity', 'SAM']
        consistent_momentum = 0.9
        rho = [1e-1, 1e-1, 2e-1]
        train_bs = [128,128]
        test_bs = [128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/'+network.lower()+str(depth)+'/test'
        args_train = parser.parse_known_args(args=[
            '--data_path', './data/ImageNet', 
            '--learning_rate', str(learning_rate[i]),
            '--network', network,
            '--depth', str(depth),
            '--dataset', dataset,
            '--minimizer', minimizer[i],
            '--rho', str(rho[i]),
            '--first_rho', str(rho[i]),
            '--second_rho', str(rho[2]),
            '--consistent_momentum', str(consistent_momentum),
            '--num_workers',str(16),
            '--train_batch_size',str(train_bs[i]),
            '--test_batch_size',str(test_bs[i]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['CSAM','GAM', 'SAM', 'ASAM', 'CSAM_Identity', 'ACSAM_Identity'], \
            f"Invalid minimizer type. Please select the correct minimizer"
        train(args_train)
