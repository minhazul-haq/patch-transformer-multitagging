# author: Mohammad Minhazul Haq
# created on: February 23, 2020

import torch
import torchvision
import os
import os.path as osp
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn as nn
from torch.utils import data
from dataset_loader import WSI_Dataset
from utils import transform_pipe_train, transform_pipe_val_test, Identity
from tensorboardX import SummaryWriter
from patch_transformation_module import PatchTransformationModule
from multitag_attention_module import MultitagAttentionModule


EPOCHS = 50
TRAIN_VAL_DIR = "/smile/local/media/Data2/ashwin/random_patches"
TRAIN_ID_LIST_FILE = "train_baseline_list.npy"
VAL_ID_LIST_FILE = "val_baseline_list.npy"
SAVED_MODEL_DIR = 'saved_model'
LOG_PATH = 'logs'
BATCH_SIZE = 32
LEARNING_RATE_RESNET = 1e-4
LEARNING_RATE_PATCH_TRANSFORMER = 1e-4
LEARNING_RATE_MULTI_TAG_ATTENTION = 1e-4
GPU_DEVICE = torch.device("cuda:0")


def validate(resnet50_model, patch_transformer, multi_tag_attention, validation_loader, loss_function):
    loss_sum = 0
    correct_sum = 0
    samples = len(validation_loader)
    softmax = torch.nn.Softmax(dim=1)

    for iter, batch in enumerate(validation_loader):
        image, label, name = batch
        image = np.squeeze(image).float()
        image = image.to(GPU_DEVICE)
        label = label.to(GPU_DEVICE)

        visual_features = resnet50_model(image)  # MxD
        patch_transformer_features = patch_transformer(visual_features)  # MxD
        prediction = multi_tag_attention(patch_transformer_features)  # 1x4

        loss = loss_function(prediction, label)
        loss_value = loss.data.cpu().numpy()
        loss_sum += loss_value.sum()

        predicted_class = torch.max(softmax(prediction), 1)[1]
        num_corrects = torch.sum(predicted_class == label).data.cpu().numpy()
        correct_sum += num_corrects

        print("iter:{0:4d}, loss:{1:.3f}, prediction/label:{2:1d}/{3:1d}".format
              (iter + 1, loss_value, predicted_class.data.cpu().numpy()[0], label.data.cpu().numpy()[0]))

    val_loss = float(loss_sum) / float(samples)
    val_accuracy = float(correct_sum) / float(samples)

    return val_loss, val_accuracy


def train():
    if not osp.exists(SAVED_MODEL_DIR):
        os.makedirs(SAVED_MODEL_DIR)

    if not osp.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    writer = SummaryWriter(log_dir=LOG_PATH)
    cudnn.enabled = True

    train_loader = data.DataLoader(WSI_Dataset(dir=TRAIN_VAL_DIR,
                                               id_list_file=TRAIN_ID_LIST_FILE,
                                               batch_size=BATCH_SIZE,
                                               transform=transform_pipe_train),
                                   batch_size=1)

    validation_loader = data.DataLoader(WSI_Dataset(dir=TRAIN_VAL_DIR,
                                                    id_list_file=VAL_ID_LIST_FILE,
                                                    batch_size=BATCH_SIZE,
                                                    transform=transform_pipe_val_test),
                                        batch_size=1)

    print("Total training wsi: " + str(len(train_loader)))
    print("Total validation wsi: " + str(len(validation_loader)))

    resnet50_model = torchvision.models.resnet50(pretrained=True)

    #remove the final fully connected layer to suite the problem
    resnet50_model.fc = Identity()

    patch_transformer = PatchTransformationModule()
    multi_tag_attention = MultitagAttentionModule()

    # if torch.cuda.device_count() > 1:
    #     print("using " + str(torch.cuda.device_count()) + "GPUs...")
    #     resnet50_model = nn.DataParallel(resnet50_model)
    #     patch_transformer = nn.DataParallel(patch_transformer)
    #     multi_tag_attention = nn.DataParallel(multi_tag_attention)

    resnet50_model.train()
    patch_transformer.train()
    multi_tag_attention.train()

    resnet50_model.to(GPU_DEVICE)
    patch_transformer.to(GPU_DEVICE)
    multi_tag_attention.to(GPU_DEVICE)

    cudnn.benchmark = True

    optimizer_resnet = torch.optim.Adam(resnet50_model.parameters(), lr=LEARNING_RATE_RESNET)
    optimizer_patch_transformer = torch.optim.Adam(patch_transformer.parameters(), lr=LEARNING_RATE_PATCH_TRANSFORMER)
    optimizer_multi_tag_attention = torch.optim.Adam(multi_tag_attention.parameters(), lr=LEARNING_RATE_MULTI_TAG_ATTENTION)

    optimizer_resnet.zero_grad()
    optimizer_patch_transformer.zero_grad()
    optimizer_multi_tag_attention.zero_grad()

    # weight = np.array([1 / float(71), 1 / float(132), 1 / float(411), 1 / float(191)])
    # weight_tensor = torch.from_numpy(weight).float().to(GPU_DEVICE)
    # mce_loss = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    mce_loss = torch.nn.CrossEntropyLoss()

    softmax = torch.nn.Softmax(dim=1)
    best_val_accuracy = 0.0

    for epoch in range(1, EPOCHS+1):
        resnet50_model.train()
        patch_transformer.train()
        multi_tag_attention.train()

        for iter, batch in enumerate(train_loader):
            image, label, name = batch
            image = np.squeeze(image).float()
            image = image.to(GPU_DEVICE)
            label = label.to(GPU_DEVICE)

            optimizer_resnet.zero_grad()
            optimizer_patch_transformer.zero_grad()
            optimizer_multi_tag_attention.zero_grad()

            visual_features = resnet50_model(image)  # MxD
            print(visual_features.shape)

            patch_transformer_features = patch_transformer(visual_features)  # MxD
            print(patch_transformer_features.shape)

            prediction = multi_tag_attention(patch_transformer_features)  # 1x4
            print(prediction.shape)

            loss = mce_loss(prediction, label)
            loss.backward()
            loss_value = loss.data.cpu().numpy()

            optimizer_resnet.step()
            optimizer_patch_transformer.step()
            optimizer_multi_tag_attention.step()

            predicted_class = torch.max(softmax(prediction), 1)[1]

            print("epoch:{0:3d}, iter:{1:4d}, loss:{2:.3f}, prediction/label:{3:1d}/{4:1d}".format
                  (epoch, iter + 1, loss_value, predicted_class.data.cpu().numpy()[0], label.data.cpu().numpy()[0]))

        #validation
        resnet50_model.eval()
        patch_transformer.eval()
        multi_tag_attention.eval()

        print("validating...")
        val_loss, val_accuracy = validate(resnet50_model, patch_transformer, multi_tag_attention, validation_loader, mce_loss)

        print("val_loss: {0:.3f}, val_accuracy: {1:.3f}".format(val_loss, val_accuracy))

        writer.add_scalar('validation loss', val_loss, epoch)
        writer.add_scalar('validation accuracy', val_accuracy, epoch)

        if (val_accuracy > best_val_accuracy):
            best_val_accuracy = val_accuracy

            print('saving best model so far...')
            torch.save(resnet50_model.state_dict(), osp.join(SAVED_MODEL_DIR, 'best_model_resnet50_' + str(epoch) + '.pth'))
            torch.save(patch_transformer.state_dict(), osp.join(SAVED_MODEL_DIR, 'best_model_patch_transformer' + str(epoch) + '.pth'))
            torch.save(multi_tag_attention.state_dict(), osp.join(SAVED_MODEL_DIR, 'best_model_multi_tag_attention_' + str(epoch) + '.pth'))


if __name__=='__main__':
    train()
