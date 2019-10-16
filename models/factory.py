import torch
import torchvision
from . import heads
from .basenetworks import ResnetBlocks, BaseNetwork, ShuffleNetV2Factory

class Shell(torch.nn.Module):
    def __init__(self, base_net, head_nets):
        super(Shell, self).__init__()

        self.base_net = base_net
        self.head_nets = torch.nn.ModuleList(head_nets)

    def io_scales(self):
        return [self.base_net.input_output_scale // (2 ** getattr(h, '_quad', 0))
                for h in self.head_nets]

    def forward(self, *args):
        x = self.base_net(*args)
        return [hn(x) for hn in self.head_nets]

def create_headnet(name, n_features):
    for head in (heads.HEADS or heads.Head.__subclasses__()):
        # LOG.debug('checking head %s matches %s', head.__name__, name)
        if not head.match(name):
            continue
        # LOG.info('selected head %s for %s', head.__name__, name)
        return head(name, n_features)

    raise Exception('unknown head to create a head network: {}'.format(name))

def model_defaults(net_cpu):
    for m in net_cpu.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # avoid numerical instabilities
            # (only seen sometimes when training with GPU)
            # Variances in pretrained models can be as low as 1e-17.
            # m.running_var.clamp_(min=1e-8)
            m.eps = 1e-4  # tf default is 0.001

            # less momentum for variance and expectation
            m.momentum = 0.01  # tf default is 0.99

def resnet_factory_from_scratch(basename, base_vision, headnames):
    resnet_factory = ResnetBlocks(base_vision)

    # input block
    use_pool = 'pool0' in basename
    conv_stride = 2
    if 'is4' in basename:
        conv_stride = 4
    if 'is1' in basename:
        conv_stride = 1
    pool_stride = 2
    if 'pool0s4' in basename:
        pool_stride = 4

    # all blocks
    blocks = [
        resnet_factory.input_block(use_pool, conv_stride, pool_stride),
        resnet_factory.block2(),
        resnet_factory.block3(),
        resnet_factory.block4(),
    ]
    if 'block5' in basename:
        blocks.append(resnet_factory.block5())

    # downsample
    if 'concat' in basename:
        for b in blocks[2:]:
            resnet_factory.replace_downsample(b)


    basenet = BaseNetwork(
        torch.nn.Sequential(*blocks),
        basename,
        resnet_factory.stride(blocks),
        resnet_factory.out_channels(blocks[-1]),
    )
    headnets = [create_headnet(h, basenet.out_features) for h in headnames if h != 'skeleton']
    net_cpu = Shell(basenet, headnets)
    model_defaults(net_cpu)
    return net_cpu


def shufflenet_factory_from_scratch(basename, base_vision, out_features, headnames):
    blocks = ShuffleNetV2Factory(base_vision).blocks()
    basenet = BaseNetwork(
        torch.nn.Sequential(*blocks),
        basename,
        input_output_scale=16,
        out_features=out_features,
    )
    headnets = [create_headnet(h, basenet.out_features) for h in headnames if h != 'skeleton']
    net_cpu = Shell(basenet, headnets)
    model_defaults(net_cpu)
    return net_cpu


def create_resnet50_pifpaf():

    base_vision = torchvision.models.resnet50(pretrained=False)

    net_cpu = resnet_factory_from_scratch('resnet50block5', base_vision, ['pif', 'paf'])

    return net_cpu

def create_shufflenet2x2_pifpaf():

    base_vision = torchvision.models.shufflenet_v2_x2_0(pretrained=False)

    net_cpu = shufflenet_factory_from_scratch('shufflenetv2x2', base_vision, 2048, ['pif', 'paf'])

    return net_cpu

def create_pifpaf(basename):
    if basename not in ['resnet50', 'shufflenetv2']:
        raise ("backbone {} must in [resnet50, shufflenetv2]".format(basename))

    if basename == 'resnet50':
        return create_resnet50_pifpaf()

    if basename == 'shufflenetv2':
        return create_shufflenet2x2_pifpaf()