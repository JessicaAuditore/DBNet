import time
import anyconfig
import torch
from tqdm import tqdm
import os
import pathlib
import glob
from pprint import pformat

from utils import WarmupPolyLR, setup_logger, cal_text_score, runningScore


class Trainer:

    def __init__(self, config, model, criterion, train_loader, validate_loader, metric_cls, post_process=None):
        config['trainer']['output_dir'] = os.path.join(str(pathlib.Path(os.path.abspath(__name__)).parent.parent),
                                                       config['trainer']['output_dir'])
        self.save_dir = config['trainer']['output_dir']
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')

        # if config['trainer']['resume_checkpoint'] == '':
        #     shutil.rmtree(self.save_dir, ignore_errors=True)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.global_step = 0
        self.start_epoch = 0
        self.config = config
        self.model = model
        self.criterion = criterion

        # logger
        self.epochs = self.config['trainer']['epochs']
        self.log_iter = self.config['trainer']['log_iter']

        anyconfig.dump(config, os.path.join(self.save_dir, 'config.yaml'))
        self.logger = setup_logger(os.path.join(self.save_dir, 'train.log'))
        self.logger_info(pformat(self.config))

        # device
        torch.manual_seed(self.config['trainer']['seed'])  # 为CPU设置随机种子
        if torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.config['trainer']['seed'])  # 为当前GPU设置随机种子
        else:
            self.with_cuda = False
            self.device = torch.device("cpu")
        self.logger_info('train with device {} and pytorch {}'.format(self.device, torch.__version__))

        # metrics and optimizer
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'), 'best_model_epoch': 0}
        self.optimizer = self._initialize('optimizer', torch.optim, model.parameters())

        # checkpoint
        if self.config['trainer']['resume_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['resume_checkpoint'], False)
            self.net_save_path_best = ''
        else:
            net_save_path_latest = os.path.join(self.checkpoint_dir, "model_latest.pth")
            if os.path.isfile(net_save_path_latest):
                self._load_checkpoint(net_save_path_latest, False)

            self.net_save_path_best = os.path.join(self.checkpoint_dir, "model_best*.pth")
            if glob.glob(self.net_save_path_best):
                self.net_save_path_best = glob.glob(self.net_save_path_best)[0]
                self._load_checkpoint(self.net_save_path_best, True)
            else:
                self.net_save_path_best = ''

        self.model.to(self.device)

        # normalize
        self.UN_Normalize = False
        for t in self.config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] == 'Normalize':
                self.normalize_mean = t['args']['mean']
                self.normalize_std = t['args']['std']
                self.UN_Normalize = True

        self.show_images_iter = self.config['trainer']['show_images_iter']
        self.train_loader = train_loader
        if validate_loader is not None:
            assert post_process is not None and metric_cls is not None
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric_cls = metric_cls
        self.train_loader_len = len(train_loader)

        # lr_scheduler
        warmup_iters = config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len
        if self.start_epoch > 1:
            self.config['lr_scheduler']['args']['last_epoch'] = (self.start_epoch - 1) * self.train_loader_len
        self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                      warmup_iters=warmup_iters, **config['lr_scheduler']['args'])

        self.logger_info(
            'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                len(self.train_loader.dataset), self.train_loader_len, len(self.validate_loader.dataset),
                len(self.validate_loader)))

        self.epoch_result = {'train_loss': 0, 'lr': 0, 'time': 0, 'epoch': 0}

    def train(self):
        # Full training logic
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            self.epoch_result = self._train_epoch(epoch)
            self._on_epoch_finish()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_text = runningScore(2)
        lr = self.optimizer.param_groups[0]['lr']

        for i, batch in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            # 数据进行转换和丢到gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            cur_batch_size = batch['img'].size()[0]

            preds = self.model(batch['img'])
            loss_dict = self.criterion(preds, batch)
            # backward
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            self.scheduler.step()

            # acc iou
            score_shrink_map = cal_text_score(preds[:, 0, :, :], batch['shrink_map'], batch['shrink_mask'],
                                              running_metric_text,
                                              thred=self.config['post_processing']['args']['thresh'])

            # loss 和 acc 记录到日志
            loss_str = 'loss: {:.4f}, '.format(loss_dict['loss'].item())
            for idx, (key, value) in enumerate(loss_dict.items()):
                loss_dict[key] = value.item()
                if key == 'loss':
                    continue
                loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
                if idx < len(loss_dict) - 1:
                    loss_str += ', '

            train_loss += loss_dict['loss']
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']

            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                self.logger_info(
                    '[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, acc: {:.4f}, iou_shrink_map: {:.4f}, {}lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step,
                                            self.log_iter * cur_batch_size / batch_time, acc, iou_shrink_map, loss_str,
                        lr, batch_time))
                batch_start = time.time()

        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self):
        self.model.eval()
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(batch, preds, is_output_polygon=self.metric_cls.is_output_polygon)
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        self.logger_info('FPS:{}'.format(total_frame / total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg

    def _on_epoch_finish(self):
        self.logger_info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))

        net_save_path_latest = '{}/model_latest.pth'.format(self.checkpoint_dir)
        self.logger_info("Saving latest checkpoint: {}".format(net_save_path_latest))
        self._save_checkpoint(self.epoch_result['epoch'], net_save_path_latest)

        recall, precision, hmean = self._eval()
        self.logger_info('test: recall: {}, precision: {}, hmean: {}'.format(recall, precision, hmean))

        if hmean > self.metrics['hmean']:
            if self.net_save_path_best != '':
                os.remove(self.net_save_path_best)

            self.metrics['train_loss'] = self.epoch_result['train_loss']
            self.metrics['hmean'] = hmean
            self.metrics['precision'] = precision
            self.metrics['recall'] = recall
            self.metrics['best_model_epoch'] = self.epoch_result['epoch']

            self.net_save_path_best = '{}/model_best_recall_{:.6f}_precision_{:.6f}_hmean_{:.6f}.pth'.format(
                self.checkpoint_dir,
                self.metrics['recall'],
                self.metrics['precision'],
                self.metrics['hmean'])
            self.logger_info("Saving best checkpoint: {}".format(self.net_save_path_best))
            self._save_checkpoint(self.epoch_result['epoch'], self.net_save_path_best)

        best_str = 'current best:'
        for k, v in self.metrics.items():
            best_str += '{}: {:.6f}, '.format(k, v)
        self.logger_info(best_str)

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger_info('{}:{}'.format(k, v))
        self.logger_info('finish train')

    def _save_checkpoint(self, epoch, file_name):
        state_dict = self.model.state_dict()
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        filename = os.path.join(self.checkpoint_dir, file_name)
        torch.save(state, filename)

    def _load_checkpoint(self, checkpoint_path, is_best):
        self.logger_info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        if is_best:
            self.metrics = checkpoint['metrics']
            self.logger_info("metrics resume from checkpoint {}".format(checkpoint_path))
            return

        self.model.load_state_dict(checkpoint['state_dict'])
        self.global_step = checkpoint['global_step']
        self.start_epoch = checkpoint['epoch']
        self.config['lr_scheduler']['args']['last_epoch'] = self.start_epoch
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']
        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        self.logger_info("resume from checkpoint {} (epoch {})".format(checkpoint_path, self.start_epoch))

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def logger_info(self, s):
        self.logger.info(s)
