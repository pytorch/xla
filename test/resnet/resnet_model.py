import torch
import torch.nn as nn
import torch.nn.functional as F
import math

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(IdentityBlock, self).__init__()

        filters1, filters2, filters3 = filters

        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY)

        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY)

        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY)

        self.relu = nn.ReLU()

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

        out += identity
        out = self.relu(out)

        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride=(2,2)):
        super(ConvBlock, self).__init__()
    
        filters1, filters2, filters3 = filters

        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY)

        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY)

        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY)

        self.shortcut_conv = nn.Conv2d(in_channels, filters3, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(filters3, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY)
        

        self.relu = nn.ReLU()

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

        shortcut = self.shortcut_conv(identity)
        shortcut = self.shortcut_bn(shortcut)

        out += shortcut
        out = self.relu(out)
        return out
 

class Resnet50(nn.Module):

	def __init__(self, num_classes=1000):
		super(Resnet50, self).__init__()
		self.conv1_pad = nn.ZeroPad2d(padding=3)
		self.conv1 = nn.Conv2d(
							in_channels=3, 
	            			out_channels=64,
	            			kernel_size=(7, 7),
	            			stride=(2, 2),
	            			padding='valid',
	            			bias=False
	        			)
		self.bn1 = nn.BatchNorm2d(64, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_DECAY, affine=True)
		self.relu1 = nn.ReLU()	
		self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

		# block_start

		self.conv_block1 = ConvBlock(in_channels=64, filters=[64,64,256], kernel_size=3, stride=(1,1))
		self.identity_block1_1 = IdentityBlock(in_channels=256, filters=[64,64,256], kernel_size=3)
		self.identity_block1_2 = IdentityBlock(in_channels=256, filters=[64,64,256], kernel_size=3)

		self.conv_block2 = ConvBlock(in_channels=256, filters=[128,128,512], kernel_size=3)
		self.identity_block2_1 = IdentityBlock(in_channels=512, filters=[128,128,512], kernel_size=3)
		self.identity_block2_2 = IdentityBlock(in_channels=512, filters=[128,128,512], kernel_size=3)
		self.identity_block2_3 = IdentityBlock(in_channels=512, filters=[128,128,512], kernel_size=3)
		
		self.conv_block3 = ConvBlock(in_channels=512, filters=[256,256,1024], kernel_size=3)
		self.identity_block3_1 = IdentityBlock(in_channels=1024, filters=[256,256,1024], kernel_size=3)
		self.identity_block3_2 = IdentityBlock(in_channels=1024, filters=[256,256,1024], kernel_size=3)
		self.identity_block3_3 = IdentityBlock(in_channels=1024, filters=[256,256,1024], kernel_size=3)
		self.identity_block3_4 = IdentityBlock(in_channels=1024, filters=[256,256,1024], kernel_size=3)
		self.identity_block3_5 = IdentityBlock(in_channels=1024, filters=[256,256,1024], kernel_size=3)

		self.conv_block4 = ConvBlock(in_channels=1024, filters=[512,512,2048], kernel_size=3)
		self.identity_block4_1 = IdentityBlock(in_channels=2048, filters=[512,512,2048], kernel_size=3)
		self.identity_block4_2 = IdentityBlock(in_channels=2048, filters=[512,512,2048], kernel_size=3)

		# Need to port this step
		#x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(2048, num_classes)
		self.softmax = nn.Softmax(dim=-1)

		# Weight initializations
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.zeros_(m.running_mean)
				nn.init.ones_(m.running_var)

	def forward(self,x):
		x = self.conv1_pad(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.max_pool(x)

		x = self.conv_block1(x)
		x = self.identity_block1_1(x)
		x = self.identity_block1_2(x)

		x = self.conv_block2(x)
		x = self.identity_block2_1(x)
		x = self.identity_block2_2(x)
		x = self.identity_block2_3(x)

		x = self.conv_block3(x)
		x = self.identity_block3_1(x)
		x = self.identity_block3_2(x)
		x = self.identity_block3_3(x)
		x = self.identity_block3_4(x)
		x = self.identity_block3_5(x)

		x = self.conv_block4(x)
		x = self.identity_block4_1(x)
		x = self.identity_block4_2(x)

		# [TODO:chandrasekhard]
		# Implement reduce_mean after porting it to functional
		# rm_axes = [2,3]
		# reduce_mean

		x = self.avgpool(x)
		x = torch.flatten(x, 1)

		x = self.fc(x)

		return x
