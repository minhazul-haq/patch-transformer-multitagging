# author: Mohammad Minhazul Haq
# created on: March 11, 2020

import torch
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils import data
from dataset_loader import WSI_Dataset
from utils import transform_pipe_val_test, Identity
from patch_transformation_module import PatchTransformationModule
from multitag_attention_module import MultitagAttentionModule


TRAIN_VAL_DIR = "/smile/local/media/Data2/ashwin/random_patches"
VAL_ID_LIST_FILE = "val_baseline_list.npy"
SAVED_MODEL_DIR = 'saved_model'
BATCH_SIZE = 32
GPU_DEVICE = torch.device("cuda:0")
BEST_MODEL_EPOCH = 15
RESNET_BEST_MODEL_PATH = "saved_model/best_model_resnet50_" + str(BEST_MODEL_EPOCH) + ".pth"
PATCH_TRANSFORMER_BEST_MODEL_PATH = "saved_model/best_model_patch_transformer" + str(BEST_MODEL_EPOCH) + ".pth"
MULTI_TAG_ATTENTION_BEST_MODEL_PATH = "saved_model/best_model_multi_tag_attention_" + str(BEST_MODEL_EPOCH) + ".pth"


def predict_and_evaluate():
    validation_loader = data.DataLoader(WSI_Dataset(dir=TRAIN_VAL_DIR,
                                                    id_list_file=VAL_ID_LIST_FILE,
                                                    batch_size=BATCH_SIZE,
                                                    transform=transform_pipe_val_test),
                                        batch_size=1)

    resnet50_model = torchvision.models.resnet50()

    # remove the final fully connected layer to suite the problem
    resnet50_model.fc = Identity()

    patch_transformer = PatchTransformationModule()
    multi_tag_attention = MultitagAttentionModule()

    resnet50_model.load_state_dict(torch.load(RESNET_BEST_MODEL_PATH))
    patch_transformer.load_state_dict(torch.load(PATCH_TRANSFORMER_BEST_MODEL_PATH))
    multi_tag_attention.load_state_dict(torch.load(MULTI_TAG_ATTENTION_BEST_MODEL_PATH))

    resnet50_model.eval()
    patch_transformer.eval()
    multi_tag_attention.eval()

    resnet50_model.to(GPU_DEVICE)
    patch_transformer.to(GPU_DEVICE)
    multi_tag_attention.to(GPU_DEVICE)

    cudnn.benchmark = True

    mce_loss = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    loss_sum = 0
    correct_sum = 0
    samples = len(validation_loader)

    for iter, batch in enumerate(validation_loader):
        image, label, name = batch
        image = np.squeeze(image).float()
        image = image.to(GPU_DEVICE)
        label = label.to(GPU_DEVICE)

        visual_features = resnet50_model(image)  # MxD
        patch_transformer_features = patch_transformer(visual_features)  # MxD
        prediction = multi_tag_attention(patch_transformer_features)  # 1x4

        loss = mce_loss(prediction, label)
        loss_value = loss.data.cpu().numpy()
        loss_sum += loss_value.sum()

        predicted_class = torch.max(softmax(prediction), 1)[1]
        num_corrects = torch.sum(predicted_class == label).data.cpu().numpy()
        correct_sum += num_corrects

        print("iter:{0:4d}, loss:{1:.3f}, prediction/label:{2:1d}/{3:1d}".format
              (iter + 1, loss_value, predicted_class.data.cpu().numpy()[0], label.data.cpu().numpy()[0]))

    val_loss = float(loss_sum) / float(samples)
    val_accuracy = float(correct_sum) / float(samples)

    print("Validation loss: " + str(val_loss))
    print("Validation accuracy: " + str(val_accuracy))


if __name__=='__main__':
    predict_and_evaluate()
