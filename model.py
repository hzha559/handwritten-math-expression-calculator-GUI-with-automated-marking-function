def network():
        #this function only run once, when the GUI starts
        print('initializing services')
        import numpy as np
        import fs
        import torchvision
        from torchvision import transforms
        import torch
        from torch.utils.data import Dataset, DataLoader
        import torch.nn as nn
        #cuda = torch.cuda.is_available()
        device = torch.device("cpu")  # check whether cpu or gpu should be used

        from torch.hub import load_state_dict_from_url 
        ########## the following is the official implementation of Resnet#########################

        __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                   'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                   'wide_resnet50_2', 'wide_resnet101_2']


        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
            'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
            'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
            'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
            'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        }


        def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
            """3x3 convolution with padding"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=dilation, groups=groups, bias=False, dilation=dilation)


        def conv1x1(in_planes, out_planes, stride=1):
            """1x1 convolution"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


        class BasicBlock(nn.Module):
            expansion = 1
            __constants__ = ['downsample']

            def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                         base_width=64, dilation=1, norm_layer=None):
                super(BasicBlock, self).__init__()
                if norm_layer is None:
                    norm_layer = nn.BatchNorm2d
                if groups != 1 or base_width != 64:
                    raise ValueError('BasicBlock only supports groups=1 and base_width=64')
                if dilation > 1:
                    raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
                # Both self.conv1 and self.downsample layers downsample the input when stride != 1
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = norm_layer(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = norm_layer(planes)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)

                return out


        class Bottleneck(nn.Module):
            expansion = 4
            __constants__ = ['downsample']

            def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                         base_width=64, dilation=1, norm_layer=None):
                super(Bottleneck, self).__init__()
                if norm_layer is None:
                    norm_layer = nn.BatchNorm2d
                width = int(planes * (base_width / 64.)) * groups
                # Both self.conv2 and self.downsample layers downsample the input when stride != 1
                self.conv1 = conv1x1(inplanes, width)
                self.bn1 = norm_layer(width)
                self.conv2 = conv3x3(width, width, stride, groups, dilation)
                self.bn2 = norm_layer(width)
                self.conv3 = conv1x1(width, planes * self.expansion)
                self.bn3 = norm_layer(planes * self.expansion)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)

                return out


        class ResNet(nn.Module):

            def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                         groups=1, width_per_group=64, replace_stride_with_dilation=None,
                         norm_layer=None):
                super(ResNet, self).__init__()
                if norm_layer is None:
                    norm_layer = nn.BatchNorm2d
                self._norm_layer = norm_layer

                self.inplanes = 64
                self.dilation = 1
                if replace_stride_with_dilation is None:
                    # each element in the tuple indicates if we should replace
                    # the 2x2 stride with a dilated convolution instead
                    replace_stride_with_dilation = [False, False, False]
                if len(replace_stride_with_dilation) != 3:
                    raise ValueError("replace_stride_with_dilation should be None "
                                     "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
                self.groups = groups
                self.base_width = width_per_group
                self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                       bias=False)
                self.bn1 = norm_layer(self.inplanes)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                               dilate=replace_stride_with_dilation[0])
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                               dilate=replace_stride_with_dilation[1])
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                               dilate=replace_stride_with_dilation[2])
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512 * block.expansion, num_classes)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

                # Zero-initialize the last BN in each residual branch,
                # so that the residual branch starts with zeros, and each residual block behaves like an identity.
                # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
                if zero_init_residual:
                    for m in self.modules():
                        if isinstance(m, Bottleneck):
                            nn.init.constant_(m.bn3.weight, 0)
                        elif isinstance(m, BasicBlock):
                            nn.init.constant_(m.bn2.weight, 0)

            def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
                norm_layer = self._norm_layer
                downsample = None
                previous_dilation = self.dilation
                if dilate:
                    self.dilation *= stride
                    stride = 1
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        conv1x1(self.inplanes, planes * block.expansion, stride),
                        norm_layer(planes * block.expansion),
                    )

                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer))

                return nn.Sequential(*layers)

            def _forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)

                return x

            # Allow for accessing forward method in a inherited class
            forward = _forward


        def _resnet(arch, block, layers, pretrained, progress, **kwargs):
            model = ResNet(block, layers, **kwargs)
            if pretrained:
                state_dict = load_state_dict_from_url(model_urls[arch],
                                                      progress=progress)
                model.load_state_dict(state_dict)
            return model





        def resnet152(pretrained=True, progress=True, **kwargs):
            r"""ResNet-152 model from
            `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

            Args:
                pretrained (bool): If True, returns a model pre-trained on ImageNet
                progress (bool): If True, displays a progress bar of the download to stderr
            """
            return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                           **kwargs)

        def resnet18(pretrained=True, **kwargs):
            """Constructs a ResNet-18 model.
            Args:
                pretrained (bool): If True, returns a model pre-trained on ImageNet
            """
            model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
            
            return model
        ############################end of the Resnet################################################################
 
        model=resnet18(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 16)#change the output classes of the model to 16
        features = model.conv1.in_channels

        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  #change the input channel of the model to 1
        model.load_state_dict(torch.load('final',map_location=torch.device('cpu')))  #load the pre-trained model
        model=model.to(device)
        model.eval()
        
        
        print('model is ready')
        
            
        class ratio_crop(object):
            #this class will change the aspect ratio of the images to 1:1 since standard input size of the Resnet is 224x224
            def __init__(self, ratio=1.0):
                self.ratio = ratio
            def __call__(self, images):
                    ratio=1.0
                    w=images.shape[1]
                    h=images.shape[0]
                    aspect_ratio=float(w)/float(h)
                    #print(images.shape,aspect_ratio)
                    
                    if aspect_ratio!=ratio:
                        dif = np.abs(h  - w)  #if w<h, pad w with white pixels
                        pad1, pad2 = int(dif // 2), int(dif - dif // 2)
                        pad = ((0, 0),(pad1, pad2),(0, 0))
                        images = np.pad(images, pad, "constant", constant_values=255)
                    return images

        transform = transforms.Compose([
            ratio_crop(1.0),
            transforms.ToPILImage(),
            transforms.Resize((56,56), interpolation=2),  #change the resolution of input images to 56x56
            transforms.ToTensor(),
        ])
        return(model,transform)
    
    
    
def recognize(model,path,transform):
        # input: the model, path of the images collected in UI, the transform function
        # this function will pass the images collected on the UI to the network and return the result
        import torch
        import cv2
        import os
        import numpy as np
        #cuda = torch.cuda.is_available()
        device = torch.device("cpu")
        print('loading images')

        expression=''  #the result expressiom
        for f in os.listdir(path):  #in the path where images are located
            name=os.path.join(path+str(f))
            #print(name)
            if "jpg" in name:  # if the file is a jpg image, open it
                im = cv2.imread(name)
                if im.all()!=0:  # if no pixel is black (indicating this is an empty grid)
                    #os.remove(name)
                    continue
                else:
                    im=transform(im)
                    #print(im.shape)
                    im=im[0]  #take only one channel of the RGB image
                    data=im.reshape(-1,1,56,56).float().to(device)
                    outputs = model(data)  #the image is reshaped and sent to the network
                    _, predicted = torch.max(outputs.data, 1)  #find the class with the highest score, which is the predicted result
                    #print("prediction",predicted)
                    symbol=predicted.item()  #extract the number from the tensor
                    
                    #print(symbol)
                    symbollist=[10,11,12,13,14,15]
                    expressionlist=['+','-','*','/','=','.']
                    for i in range(len(symbollist)):
                        if symbollist[i]==symbol:
                            expression+=expressionlist[i]
                            break
                        elif symbol<10:  #if the predicted number is between 0 and 9, which doesn't need to be translated
                            expression+=str(symbol)
                            break
                        else:
                            expression+=''
        return(expression)  #the expression in string form