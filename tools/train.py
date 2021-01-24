import os
import anyconfig

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    import sys
    import pathlib

    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))

    from models import build_model, build_loss
    from data_loader import get_dataloader
    from utils import Trainer
    from utils import get_post_processing
    from utils import get_metric

    config = anyconfig.load(open('config.yaml', 'rb'))
    train_loader = get_dataloader(config['dataset']['train'])
    validate_loader = get_dataloader(config['dataset']['validate'])
    criterion = build_loss(config['loss']).cuda()
    model = build_model(config['arch'])
    post_p = get_post_processing(config['post_processing'])
    metric = get_metric(config['metric'])

    trainer = Trainer(config=config,
                      model=model,
                      criterion=criterion,
                      train_loader=train_loader,
                      post_process=post_p,
                      metric_cls=metric,
                      validate_loader=validate_loader)
    trainer.train()


if __name__ == '__main__':
    main()
