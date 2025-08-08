# NOTE 在 PyTorch Lightning 2.0 版本中，pl.Trainer.add_argparse_args 这个用于自动向命令行解析器添加 Trainer 参数的函数已经被移除了。
# 同样，pl.Trainer.from_argparse_args 也被移除了。
# 本文件应由easy_train.sh自动生成，但原始文件不兼容lightning2.5.2标准，因此需要手动调整参数解析和Trainer（accelerator/devices 等）实例化方法。

import argparse
import model as M
import nnue_dataset
import pytorch_lightning as pl
import features
import os
import sys
import torch
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

def make_data_loaders(train_filename, val_filename, feature_set, num_workers, batch_size, filtered, random_fen_skipping, wld_filtered, main_device, epoch_size, val_size):
  # Epoch and validation sizes are arbitrary
  features_name = feature_set.name
  train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filename, batch_size, num_workers=num_workers,
                                                   filtered=filtered, random_fen_skipping=random_fen_skipping, wld_filtered=wld_filtered, device=main_device)
  val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filename, batch_size, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, wld_filtered=wld_filtered, device=main_device)
  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  return train, val

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_gpus(gpus_str):
    # accepts formats like "0," or "0,1," or "0"
    if not gpus_str:
        return None
    parts = [p for p in gpus_str.split(',') if p.strip() != '']
    try:
        ids = [int(p) for p in parts]
        return ids if ids else None
    except ValueError:
        return None

def main():
  parser = argparse.ArgumentParser(description="Trains the network.")
  # 训练/验证数据
  parser.add_argument("train", help="Training data (.bin or .binpack)")
  parser.add_argument("--validation-data", dest='val', default=None, help="Validation data (.bin or .binpack)")
  parser.add_argument("--validation-dataset", dest='val', help=argparse.SUPPRESS)  # 兼容别名

  # 训练超参与数据加载
  parser.add_argument("--lambda", default=1.0, type=float, dest='lambda_', help="lambda=1.0 = eval, lambda=0.0 = result.")
  parser.add_argument("--gamma", default=0.992, type=float, dest='gamma', help="LR multiplicative decay per epoch.")
  parser.add_argument("--lr", default=8.75e-4, type=float, dest='lr', help="Initial learning rate.")
  parser.add_argument("--num-workers", default=1, type=int, dest='num_workers', help="Data loader workers (binpack works best).")
  parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size', help="Positions per batch. Default GPU=16384, CPU=128.")
  parser.add_argument("--threads", default=-1, type=int, dest='threads', help="Torch intra-op threads.")
  parser.add_argument("--seed", default=42, type=int, dest='seed', help="Torch seed.")
  parser.add_argument("--smart-fen-skipping", action='store_true', dest='smart_fen_skipping_deprecated', help=argparse.SUPPRESS)
  parser.add_argument("--no-smart-fen-skipping", action='store_true', dest='no_smart_fen_skipping', help="Disable smart fen skipping.")
  parser.add_argument("--no-wld-fen-skipping", action='store_true', dest='no_wld_fen_skipping', help="Disable WLD fen skipping.")
  parser.add_argument("--random-fen-skipping", default=3, type=int, dest='random_fen_skipping', help="Skip k on average before using 1.")
  parser.add_argument("--resume-from-model", dest='resume_from_model', help=".pt checkpoint to resume from.")
  parser.add_argument("--network-save-period", type=int, default=20, dest='network_save_period', help="Epochs between snapshots.")
  parser.add_argument("--save-last-network", type=str2bool, default=True, dest='save_last_network', help="Always save last network.")
  parser.add_argument("--epoch-size", type=int, default=100000000, dest='epoch_size', help="Positions per epoch.")
  parser.add_argument("--validation-size", type=int, default=1000000, dest='validation_size', help="Positions per validation step.")

  # 兼容 easy_train 传入的 Trainer/脚本参数
  parser.add_argument("--max_epoch", type=int, default=10, dest='max_epoch', help="Max epochs (compat).")
  parser.add_argument("--default_root_dir", type=str, default=None, dest='default_root_dir', help="Logs/checkpoints root.")
  parser.add_argument("--gpus", type=str, default=None, dest='gpus', help="GPU id list string, e.g. '0,' or '0,1,'")
  parser.add_argument("--detect_anomaly", type=str2bool, default=False, dest='detect_anomaly', help="Enable autograd anomaly detection.")
  parser.add_argument("--early-fen-skipping", type=int, default=-1, dest='early_fen_skipping', help="Ignored; for compatibility.")
  parser.add_argument("--start-lambda", type=float, default=None, dest='start_lambda', help="Compatibility only; not used here.")
  parser.add_argument("--end-lambda", type=float, default=None, dest='end_lambda', help="Compatibility only; not used here.")

  features.add_argparse_args(parser)

  # 允许保留未知参数，避免因外部新参数导致崩溃
  args, unknown = parser.parse_known_args()

  if not os.path.exists(args.train):
    raise Exception('{0} does not exist'.format(args.train))
  # 验证集：如果未提供，就退化为使用训练集（或直接报错）
  if args.val is None:
    args.val = args.train
  if not os.path.exists(args.val):
    raise Exception('{0} does not exist'.format(args.val))

  feature_set = features.get_feature_set_from_name(args.features)

  if args.resume_from_model is None:
    nnue = M.NNUE(feature_set=feature_set, lambda_=args.lambda_, gamma=args.gamma, lr=args.lr)
  else:
    nnue = torch.load(args.resume_from_model)
    nnue.set_feature_set(feature_set)
    nnue.lambda_ = args.lambda_
    nnue.gamma = args.gamma
    nnue.lr = args.lr

  print("Feature set: {}".format(feature_set.name))
  print("Num real features: {}".format(feature_set.num_real_features))
  print("Num virtual features: {}".format(feature_set.num_virtual_features))
  print("Num features: {}".format(feature_set.num_features))
  print("Training with {} validating with {}".format(args.train, args.val))

  pl.seed_everything(args.seed)
  print("Seed {}".format(args.seed))

  batch_size = args.batch_size if args.batch_size > 0 else 16384
  print('Using batch size {}'.format(batch_size))
  print('Smart fen skipping: {}'.format(not args.no_smart_fen_skipping))
  print('WLD fen skipping: {}'.format(not args.no_wld_fen_skipping))
  print('Random fen skipping: {}'.format(args.random_fen_skipping))

  if args.threads > 0:
    print('limiting torch to {} threads.'.format(args.threads))
    t_set_num_threads(args.threads)

  logdir = args.default_root_dir if args.default_root_dir else 'logs/'
  print('Using log dir {}'.format(logdir), flush=True)

  tb_logger = pl_loggers.TensorBoardLogger(logdir)
  checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=args.save_last_network, every_n_epochs=args.network_save_period, save_top_k=-1)

  # 解析 GPU 设置，Lightning 2.x 风格
  device_ids = parse_gpus(args.gpus)
  if device_ids:
    accelerator = 'gpu'
    devices = device_ids
  else:
    accelerator = 'cpu'
    devices = 1

  trainer = pl.Trainer(
      max_epochs=args.max_epoch,
      default_root_dir=logdir,
      accelerator=accelerator,
      devices=devices,
      detect_anomaly=args.detect_anomaly,
      callbacks=[checkpoint_callback],
      logger=tb_logger
  )

  # 推断主设备
  try:
    main_device = trainer.strategy.root_device
  except Exception:
    if accelerator == 'gpu' and isinstance(devices, list) and len(devices) > 0:
      main_device = torch.device(f'cuda:{devices[0]}')
    else:
      main_device = torch.device('cpu')

  nnue.to(device=main_device)

  print('Using c++ data loader')
  train, val = make_data_loaders(
    args.train,
    args.val,
    feature_set,
    args.num_workers,
    batch_size,
    not args.no_smart_fen_skipping,
    args.random_fen_skipping,
    not args.no_wld_fen_skipping,
    main_device,
    args.epoch_size,
    args.validation_size)

  trainer.fit(nnue, train, val)

  with open(os.path.join(logdir, 'training_finished'), 'w'):
    pass

if __name__ == '__main__':
  main()
  if sys.platform == "win32":
    os.system(f'wmic process where processid="{os.getpid()}" call terminate >nul')