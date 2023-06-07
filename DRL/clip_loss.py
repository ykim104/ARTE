import torch
import torch.nn as nn
from torchvision import models, transforms
import CLIP_.clip as clip
import collections

def distance_metrics(xs_conv_features, ys_conv_features):
    #for x_conv, y_conv in zip(xs_conv_features, ys_conv_features):
    #    print("x conv shpae: " , torch.square(x_conv - y_conv).mean(1).mean(1).mean(1).shape) 
    
    ## if "RN" in clip model name
    return [torch.square(x_conv - y_conv).mean(1).mean(1).mean(1) for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]

    #return [1-torch.cosine_similarity(x_conv, y_conv, dim=1).mean(1).mean(1) for x_conv, y_conv in
    #            zip(xs_conv_features, ys_conv_features)]


class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps



model, clip_preprocess = clip.load(
            "RN101", "cuda", jit=False)
visual_model = model.visual
layers = list(model.visual.children())
init_layers = torch.nn.Sequential(*layers)[:8]

normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

#visual_encoder = CLIPVisualEncoder(model)


def get_clip_conv_loss(canvas, target, mode="train"):
    """
    Parameters
    ----------
    sketch: Torch Tensor [B, C, H, W]
    target: Torch Tensor [B, C, H, W]
    """

    # normalize
    x, y = [normalize_transform(canvas)], [normalize_transform(target)]

    xs = torch.cat(x, dim=0)
    ys = torch.cat(y, dim=0)


    xs_fc_features, xs_conv_features = forward_inspection_clip_resnet(
        xs.contiguous())
    ys_fc_features, ys_conv_features = forward_inspection_clip_resnet(
        ys.detach())

    # Geometric Loss: L2 distance between intermediate level activations of CLIP
    conv_loss = distance_metrics(xs_conv_features,ys_conv_features)

    conv_loss_weights = [0.8, 0.8, 1.0, 1.0, 0.8]
    for i in range(len(conv_loss)):
        conv_loss[i]*=conv_loss_weights[i]
    #print("conv_loss: ", sum(conv_loss))


    # Semantic Loss: cosine distance
    fc_loss = [1 - torch.cosine_similarity(xs_fc_features,
                    ys_fc_features, dim=1)]
    for i in range(len(fc_loss)):
        fc_loss[i]*=0.5 #0.1 #.mean()
    #print("fc loss: ", sum(fc_loss))

    return sum(fc_loss) + sum(conv_loss) #conv_loss[0]


def forward_inspection_clip_resnet(x):
    def stem(m, x):
        #print(m)
        for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
            x = m.relu(bn(conv(x)))
        x = m.avgpool(x)
        return x
    x = x.type(visual_model.conv1.weight.dtype)
    x = stem(visual_model, x)
    x1 = layers[8](x)
    x2 = layers[9](x1)
    x3 = layers[10](x2)
    x4 = layers[11](x3)
    y = layers[12](x4)
    return y, [x, x1, x2, x3, x4]
    