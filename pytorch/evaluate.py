# -*- coding:utf-8 -*-
import argparse
import logging
import os

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
import model.net as net
import model.data_loader as data_loader
import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--datafile', default=None)
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--data_dir', default='data/original', help='hehe')


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean

if __name__ == '__main__':
    args = parser.parse_args()

    assert args.datafile is not None, "Must input --data"
    size = build_dataset.SIZE
    img = Image.open(args.datafile)
    img = img.resize((size, size), Image.BILINEAR)
    img = img.convert('L')
    input_tensor = transforms.ToTensor()(img)

    assert os.path.isdir(args.data_dir), "no dataset at {}".format(args.data_dir)

    train_dir = os.path.join(args.data_dir, 'train')
    train_dir.replace('\\', '/')

    train_filenames, val_filenames, category_dict = build_dataset.split_data(args.data_dir, train_dir)

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()     # use GPU is available

    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    model.eval()

    if params.cuda:
        input_tensor = input_tensor.cuda(async=True)
    input_tensor = input_tensor.view(1, 1, size, size)
    input_tensor = Variable(input_tensor)

    pred = model(input_tensor).data.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    print("{img} is predicted for {pred}".
          format(img=args.datafile, pred=category_dict[pred[0]]))


