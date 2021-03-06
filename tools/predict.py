import os
import time
import cv2
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data_loader import get_transforms
from models import build_model
from utils.post_processing import get_post_processing


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class Pytorch_model:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, img_path: str, is_output_polygon=True, short_size: int = 1024):
        '''
        对传入的图像进行预测，支持图像地址,opencv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = resize_image(img, short_size)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t = time.time() - start
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t


def save_depoly(model, input, save_path):
    traced_script_model = torch.jit.trace(model, input)
    traced_script_model.save(save_path)


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet')
    parser.add_argument('--model_path',
                        default=r'C:\Users\94806\Desktop\output\model_best_recall_0.612750_precision_0.778183_hmean_0.685628.pth',
                        type=str)
    parser.add_argument('--input_folder', default='../test/input', type=str, help='img path for predict')
    parser.add_argument('--output_folder', default='../test/output', type=str, help='img path for output')
    parser.add_argument('--thre', default=0.3, type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--save_result', action='store_true', help='save box and score to txt file')
    args = parser.parse_args()
    return args


def predict(model, img_path: str, is_output_polygon=False, short_size: int = 1024):
    device = torch.device("cuda:0")

    img = cv2.imread(img_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    img = resize_image(img, short_size)

    checkpoint = torch.load('C:/Users/94806/Desktop/model_best_recall_0.558953_precision_0.769060_hmean_0.647386.pth',
                            map_location=device)
    config = checkpoint['config']

    transform = []
    for t in config['dataset']['train']['dataset']['args']['transforms']:
        if t['type'] in ['ToTensor', 'Normalize']:
            transform.append(t)
    transform = get_transforms(transform)

    tensor = transform(img)
    tensor = tensor.unsqueeze_(0)

    tensor = tensor.cuda()
    batch = {'shape': [(h, w)]}
    with torch.no_grad():
        torch.cuda.synchronize(device)
        start = time.time()
        preds = model(tensor)
        torch.cuda.synchronize(device)

        post_process = get_post_processing(config['post_processing'])

        box_list, score_list = post_process(batch, preds, is_output_polygon=is_output_polygon)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            if is_output_polygon:
                idx = [x.sum() > 0 for x in box_list]
                box_list = [box_list[i] for i, v in enumerate(idx) if v]
                score_list = [score_list[i] for i, v in enumerate(idx) if v]
            else:
                idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        t = time.time() - start
    return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t


if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox, save_result, get_file_list

    args = init_args()
    print(args)

    model = Pytorch_model(args.model_path, post_p_thre=args.thre, gpu_id=0)

    # input = torch.rand(1, 3, 736, 736).cuda()
    # save_depoly(model.model.eval(), input, "dbnet.pt")
    #
    # torch.save(model, './model.pth')
    # model = torch.jit.load("dbnet.pt").cuda()

    for img_path in tqdm(get_file_list(args.input_folder, p_postfix=['.jpg'])):
        preds, boxes_list, score_list, t = model.predict(img_path, is_output_polygon=False)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
        # if args.show:
        #     show_img(preds)
        #     show_img(img, title=os.path.basename(img_path))
        #     plt.show()
        # 保存结果到路径
        os.makedirs(args.output_folder, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder, img_path.stem + '_result.jpg')
        pred_path = os.path.join(args.output_folder, img_path.stem + '_pred.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        cv2.imwrite(pred_path, preds * 255)
        save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)
