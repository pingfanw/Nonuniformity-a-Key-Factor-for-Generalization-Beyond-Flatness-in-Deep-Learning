from models.cifar import (alexnet, densenet, resnet,
                          vgg16_bn, vgg19_bn, wrn, preresnet, mlp, pyramidnet)
# from models.imagenet import (resnext50, resnext101,
#                               resnext152)


def get_network_cifar(network, **kwargs):

    networks = {
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'wrn': wrn,
        'preresnet': preresnet,
        'mlp': mlp,
        'pyramidnet': pyramidnet
    }
    # print(networks[network])
    return networks[network](**kwargs)

