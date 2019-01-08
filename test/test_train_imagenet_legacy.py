import test_utils

FLAGS = test_utils.parse_common_options(
    datadir='/tmp/imagenet', batch_size=128, num_epochs=15, target_accuracy=0.0)

from common_utils import TestCase, run_tests
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla_py.utils as xu
import torch_xla_py.xla_model as xm
import unittest


def _cross_entropy_loss_eval_fn(cross_entropy_loss):

  def eval_fn(output, target):
    loss = cross_entropy_loss(output, target).item()
    # Get the index of the max log-probability.
    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(target.view_as(pred)).sum().item()
    return loss, correct

  return eval_fn


def train_imagenet():
  print('==> Preparing data..')
  if FLAGS.fake_data:
    train_loader = xu.SampleGenerator(
        data=torch.zeros(FLAGS.batch_size, 3, 224, 224),
        target=torch.zeros(FLAGS.batch_size, dtype=torch.int64),
        sample_count=1200000 // FLAGS.batch_size)
    test_loader = xu.SampleGenerator(
        data=torch.zeros(FLAGS.batch_size, 3, 224, 224),
        target=torch.zeros(FLAGS.batch_size, dtype=torch.int64),
        sample_count=50000 // FLAGS.batch_size)
  else:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.datadir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers)
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.datadir, 'val'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers)

  torch.manual_seed(42)

  print('==> Building model..')
  momentum = 0.9
  lr = 0.1
  assert FLAGS.num_cores == 1
  log_interval = 10

  model = torchvision.models.resnet50()
  inputs = torch.zeros(FLAGS.batch_size, 3, 224, 224)
  xla_model = xm.XlaModel(model, [inputs])
  optimizer = optim.SGD(
      xla_model.parameters_list(), lr=lr, momentum=momentum, weight_decay=5e-4)

  cross_entropy_loss = nn.CrossEntropyLoss()

  accuracy = None
  for epoch in range(1, FLAGS.num_epochs + 1):
    # Training loop for epoch.
    start_time = time.time()
    processed = 0
    for batch_idx, (data, target) in enumerate(train_loader):
      if data.size()[0] != FLAGS.batch_size:
        break
      optimizer.zero_grad()
      y = xla_model(data)
      y[0].requires_grad = True
      loss = cross_entropy_loss(y[0], target)
      loss.backward()
      xla_model.backward(y)
      optimizer.step()
      processed += FLAGS.batch_size
      if batch_idx % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss: {:.6f}\tSamples/sec: {:.1f}'.format(
                  epoch, processed,
                  len(train_loader) * FLAGS.batch_size,
                  100. * batch_idx / len(train_loader), loss,
                  processed / (time.time() - start_time)))

    # Eval loop for epoch.
    start_time = time.time()
    correct_count = 0
    test_loss = 0
    count = 0
    for batch_idx, (data, target) in enumerate(test_loader):
      if data.size()[0] != FLAGS.batch_size:
        break
      y = xla_model(data)
      test_loss += loss_fn(y[0], target).sum().item()
      pred = y[0].max(1, keepdim=True)[1]
      correct_count += pred.eq(target.view_as(pred)).sum().item()
      count += FLAGS.batch_size

    test_loss /= count
    accuracy = 100.0 * correct_count / count
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), '
          'Samples/sec: {:.1f}\n'.format(test_loss, correct_count, count,
                                         accuracy,
                                         count / (time.time() - start_time)))
    # Debug metric dumping.
    if FLAGS.metrics_debug:
      print(torch_xla._XLAC._xla_metrics_report())
  return accuracy


class TrainImageNet(TestCase):

  def tearDown(self):
    super(TrainImageNet, self).tearDown()

  def test_accurracy(self):
    self.assertGreaterEqual(train_imagenet(), FLAGS.target_accuracy)


# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
run_tests()
