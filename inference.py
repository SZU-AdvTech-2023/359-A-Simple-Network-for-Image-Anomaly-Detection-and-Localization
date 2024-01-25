import torch
import torch.nn as nn
from torchvision import transforms
import os
import logging
from common import *
from PIL import Image
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import math
from main import net
import backbones


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d' % (i + 1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        # self.apply(init_weight)

    def forward(self, x):
        #print("Dis x shape", x.shape)
        x = self.body(x)
        x = self.tail(x)
        return x


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                                s + 2 * padding - 1 * (self.patchsize - 1) - 1
                        ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x


class Projection(torch.nn.Module):

    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc",
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn",
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        # self.apply(init_weight)

    def forward(self, x):

        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores, features):

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features)
            features = features.to(self.device).permute(0, 3, 1, 2)
            if self.target_size[0] * self.target_size[1] * features.shape[0] * features.shape[1] >= 2**31:
                subbatch_size = int((2**31-1) / (self.target_size[0] * self.target_size[1] * features.shape[1]))
                interpolated_features = []
                for i_subbatch in range(int(features.shape[0] / subbatch_size + 1)):
                    subfeatures = features[i_subbatch*subbatch_size:(i_subbatch+1)*subbatch_size]
                    subfeatures = subfeatures.unsuqeeze(0) if len(subfeatures.shape) == 3 else subfeatures
                    subfeatures = F.interpolate(
                        subfeatures, size=self.target_size, mode="bilinear", align_corners=False
                    )
                    interpolated_features.append(subfeatures)
                features = torch.cat(interpolated_features, 0)
            else:
                features = F.interpolate(
                    features, size=self.target_size, mode="bilinear", align_corners=False
                )
            features = features.cpu().numpy()

        return [
            torch.from_numpy(ndimage.gaussian_filter(patch_score, sigma=self.smoothing))
            for patch_score in patch_scores
        ]




class simplenet(nn.Module):
    def __init__(self,
                 backbone,
                 layers_to_extract_from,
                 device,
                 input_shape,
                 pretrain_embed_dimension,  # 1536
                 target_embed_dimension,  # 1536
                 patchsize=3,  # 3
                 patchstride=1,
                 embedding_size=None,  # 256
                 dsc_layers=2,  # 2
                 dsc_hidden=None,  # 1024
                 train_backbone=False,
                 pre_proj=1,  # 1
                 proj_layer_type=0,
                 checkpoint_path=None,
                 **kwargs, ):
        super(simplenet, self).__init__()
        self.backbone = backbones.load(backbone).to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.forward_modules = torch.nn.ModuleDict({})
        self.feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = self.feature_aggregator.feature_dimensions(input_shape)
        self.preprocessing = Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.target_embed_dimension = target_embed_dimension
        self.preadapt_aggregator = Aggregator(
            target_dim=target_embed_dimension
        )
        self.pre_projection = Projection(target_embed_dimension, target_embed_dimension, pre_proj,
                                         proj_layer_type)
        #print(self.pre_projection)
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.anomaly_segmentor = RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def forward(self, images):
        batchsize = images.shape[0]
        _ = self.feature_aggregator.eval()
        with torch.no_grad():
            features = self.feature_aggregator(images)
            features = [features[layer] for layer in self.layers_to_extract_from]

            for i, feat in enumerate(features):
                if len(feat.shape) == 3:
                    B, L, C = feat.shape
                    features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)
            features = [
                self.patch_maker.patchify(x, return_spatial_info=True) for x in features
            ]
            # print("features shape: ", features[0].shape)
            patch_shapes = [x[1] for x in features]
            features = [x[0] for x in features]
            ref_num_patches = patch_shapes[0]
            # self.Resize(features, patch_shapes, ref_num_patches)
            for i in range(1, len(features)):
                _features = features[i]
                patch_dims = patch_shapes[i]

                # TODO(pgehler): Add comments
                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )
                _features = _features.permute(0, -3, -2, -1, 1, 2)
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])
                _features = F.interpolate(
                    _features.unsqueeze(1),
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                _features = _features.squeeze(1)
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )
                _features = _features.permute(0, -2, -1, 1, 2, 3)
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                features[i] = _features
            features = [x.reshape(-1, *x.shape[-3:]) for x in features]

            features = self.preprocessing(features)  # pooling each feature to same channel and stack together

            features = self.preadapt_aggregator(features)  # further pooling
            
            features = self.pre_projection(features)
            patch_scores = image_scores = -self.discriminator(features)
        patch_scores = patch_scores.cpu().numpy()
        image_scores = image_scores.cpu().numpy()

        image_scores = self.patch_maker.unpatch_scores(
            image_scores, batchsize=batchsize
        )
        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
        image_scores = self.patch_maker.score(image_scores)

        patch_scores = self.patch_maker.unpatch_scores(
            patch_scores, batchsize=batchsize
        )
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
        features = features.reshape(batchsize, scales[0], scales[1], -1)
        masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)
        #print(type(masks[0]), type(features[0]))
        return masks[0]
        
    def Resize(self, features, patch_shapes, ref_num_patches):
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        return features

    def load_checkpoint(self, checkpoint):
        model = torch.load(checkpoint)
        # self.feature_aggregator.load_state_dict(model["feature_aggregator"])
        # self.preprocessing.load_state_dict(model["preprocessing"])
        # self.preadapt_aggregator.load_state_dict(model["preadapt_aggregator"])
        self.pre_projection.load_state_dict(model["pre_projection"])
        self.discriminator.load_state_dict(model["discriminator"])


def transform(image):
    data_transform = transforms.Compose([transforms.Resize((288, 288)),
                                         transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transform(image)


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


#
def _inference_one(file, model, device, save_path):
    time_cost = 0
    image = Image.open(file)
    img = image

    image = transforms.Grayscale(num_output_channels=3)(image)

    image = transform(image).to(device)
    image = torch.unsqueeze(image, 0)
    print("image shape: ", image.shape)
    # model.eval()
    start = timer()
    mask = model(image)
    end = timer()
    with open('time.txt', 'w') as file:
        file.write(f'推理一张图片用时：{end - start}')
    print("推理一张图片用时：", end - start)
    time_cost += (end - start)
    #out = torch.squeeze(torch.squeeze(out, 0), 0)

    #heatmap = out.cpu().detach().numpy()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()
    plt.savefig("./results/test.png")


def _inference_batch():
    pass


def _inference_files(file, gpu):
    """
    """
    pass


def get_simplenet(checkpoint_path=None):
    sim = simplenet(backbone="wideresnet50",
                    layers_to_extract_from=["layer2", "layer3"],
                    device="cuda:1",
                    input_shape=(3, 288, 288),
                    pretrain_embed_dimension=1536,
                    target_embed_dimension=1536,
                    patchsize=3,  # 3
                    patchstride=1,
                    embedding_size=None,  # 256
                    dsc_layers=2,  # 2
                    dsc_hidden=None,  # 1024
                    train_backbone=False,
                    pre_proj=1,  # 1
                    proj_layer_type=0,
                    checkpoint_path=checkpoint_path, )
    return sim


def inference(file, model_path, device, save_path=None):
    """
    :func:
    :param file:
    --------func: the files need to inference, it can be a path or png, jpg. bmp file
    --------Option: the files need to inference, it can be a path or png, jpg. bmp file
    --------default: the files need to inference, it can be a path or png, jpg. bmp file
    :param model_path:
    --------func: the checkpoint path of model
    --------Option: the files need to inference, it can be a path or png, jpg. bmp file
    --------default: the files need to inference, it can be a path or png, jpg. bmp file
    :param gpu:
     --------func: the files need to inference, it can be a path or png, jpg. bmp file
    --------Option: the files need to inference, it can be a path or png, jpg. bmp file
    --------default: the files need to inference, it can be a path or png, jpg. bmp file
    :param save_path: the path to save results, if no need to save, it is None
    --------func: the files need to inference, it can be a path or png, jpg. bmp file
    --------Option: the files need to inference, it can be a path or png, jpg. bmp file
    --------default: the files need to inference, it can be a path or png, jpg. bmp file
    --------
    :return:
    """
    if model_path is not None and os.path.exists(model_path):
        model = get_simplenet(model_path).to(device)
        m = get_simplenet().to(device)
        #torch.save(model, "./model.pth")
        torch.save({"state_dict":model.state_dict()}, "./model.ckpt")
        m.load_state_dict(torch.load("./model.ckpt")['state_dict'])
        # model = get_simplenet()
        # model.load_state_dict(torch.load(model_path))
    else:
        raise Exception("the model path is empty OR the model hasn't been set.....")

    if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".bmp"):  # inference one image
        _inference_one(file, m, device, save_path=None)
    elif os.path.exists(file):  # inference images in a path
        pass
    else:
        raise Exception("the tested file is empty....")

    if save_path is not None:
        pass


if __name__ == "__main__":
    device = set_torch_device([1])
    inference("./pictures/bottle.png", "./simplenet.ckpt", device, "./results")