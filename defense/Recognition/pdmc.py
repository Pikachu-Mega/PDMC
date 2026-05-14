import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import pformat
import yaml
import logging
import time
from defense.base import defense

from torch.utils.data import DataLoader, RandomSampler

from copy import deepcopy

from utils.trainer_cls import BackdoorModelTrainer, Metric_Aggregator, ModelTrainerCLS, ModelTrainerCLS_v2, \
    PureCleanModelTrainer, general_plot_for_epoch
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model, partially_load_state_dict
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from utils.mc import curve_models
from utils.mc.connectivity import testCurve
from utils.curve import curve_opt
from utils.sam_tool.bypass_bn import enable_running_stats, disable_running_stats
from utils.permutation_utils import permutation
# ==== CSV helpers ====
import csv

def _csv_path(args):
    # 统一落在 checkpoint_save 目录下
    os.makedirs(args.checkpoint_save, exist_ok=True)
    return os.path.join(args.checkpoint_save, "metrics_steps.csv")

def csv_write_header_if_needed(args):
    p = _csv_path(args)
    if not os.path.exists(p):
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "time", "round", "phase", "epoch",
                "clean_acc", "asr", "ra",
                "c_acc",                      # 若你与 clean_acc 含义一致，可与其相同
                "time_sec", "peak_mem_MB",
                "fwd", "bwd"
            ])

def csv_write_row(args, round_idx, phase, epoch,
                  clean_acc, asr, ra,
                  c_acc, time_sec, peak_mem_MB, fwd, bwd):
    csv_write_header_if_needed(args)
    with open(_csv_path(args), "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            round_idx, phase, epoch,
            f"{clean_acc:.3f}" if clean_acc is not None else "",
            f"{asr:.3f}" if asr is not None else "",
            f"{ra:.3f}" if ra is not None else "",
            f"{c_acc:.3f}" if c_acc is not None else "",
            f"{time_sec:.3f}" if time_sec is not None else "",
            f"{peak_mem_MB:.1f}" if peak_mem_MB is not None else "",
            fwd if fwd is not None else "",
            bwd if bwd is not None else "",
        ])
# ============================================

# ==== Logging helpers for compute metrics ====
PASS_COUNTER = {"fwd": 0, "bwd": 0}

def reset_pass_counters():
    PASS_COUNTER["fwd"] = 0
    PASS_COUNTER["bwd"] = 0

def inc_fwd(n=1):  PASS_COUNTER["fwd"] += n
def inc_bwd(n=1):  PASS_COUNTER["bwd"] += n

def step_begin(tag: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    logging.info(f"[STEP-BEGIN] {tag}")
    reset_pass_counters()   # 进入一个“步骤”时重置 fwd/bwd 统计
    return t0

def step_end(tag: str, t0: float):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    else:
        peak_mem = -1
    dt = time.time() - t0
    logging.info(f"[STEP-END] {tag} | time={dt:.3f}s | peak_mem_MB={peak_mem:.1f} | fwd={PASS_COUNTER['fwd']} | bwd={PASS_COUNTER['bwd']}")
    return dt, peak_mem, PASS_COUNTER["fwd"], PASS_COUNTER["bwd"]
# =============================================

def get_curve_class(args):
    if args.model == 'PreResNet110':
        net = getattr(curve_models, 'PreResNet110')
    elif args.model == 'preactresnet18':
        net = getattr(curve_models, 'PreActResNet18Arc')
    elif args.model == 'resnet50':
        net = getattr(curve_models, 'ResNet18Arc')

    else:
        raise SystemError('NO valid model match in function generate_cls_model!')
    return net

def oneEpochTrain(args, model, train_data_loader, criterion, optimizer, scheduler, device, regular):
    batch_loss = []
    # —— per-epoch 显存统计起点 ——
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    reset_pass_counters()

    for i, (inputs, labels, *additional_info) in enumerate(tqdm(train_data_loader)):  # type: ignore
        model.train()
        model.to(device)
        inputs, labels = inputs.to(device), labels.to(device)

        # forward
        outputs = model(inputs); inc_fwd()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        batch_loss.append(loss.item() * labels.size(0))
        if regular is not None:
            loss += regular(model)

        # backward + step
        loss.backward(); inc_bwd()
        optimizer.step()

    one_epoch_loss = sum(batch_loss) / len(train_data_loader.dataset)
    if args.lr_scheduler == 'ReduceLROnPlateau' and scheduler is not None:
        scheduler.step(one_epoch_loss)
    elif args.lr_scheduler == 'CosineAnnealingLR' and scheduler is not None:
        scheduler.step()

    end_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    else:
        peak_mem = -1

    # 写入 CSV（epoch级统计；phase 在调用处传入）
    try:
        csv_write_row(args,
                      getattr(args, "_round_idx", -1),        # 由调用处设置
                      getattr(args, "_phase_tag", "epoch"),   # 由调用处设置
                      getattr(args, "_epoch_idx", None),      # 由调用处设置
                      clean_acc=None, asr=None, ra=None,      # epoch 内不评估效果指标
                      c_acc=None,
                      time_sec=end_time - start_time,
                      peak_mem_MB=peak_mem,
                      fwd=PASS_COUNTER['fwd'],
                      bwd=PASS_COUNTER['bwd'])
    except Exception as e:
        logging.info(f"[CSV] skip epoch row: {e}")

    logging.info(
        f"one epoch training part done, use time = {end_time - start_time} s | "
        f"peak_mem_MB={peak_mem:.1f} | fwd={PASS_COUNTER['fwd']} | bwd={PASS_COUNTER['bwd']}"
    )
    return one_epoch_loss



def reset_bn_stats(model, data_loader):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None  # use simple average
            m.reset_running_stats()
    model.train().cuda()
    with torch.no_grad():
        for images, _ in data_loader:
            output = model(images.cuda())


first_stage_curve_acc_result = []
second_stage_curve_acc_result = []

first_stage_curve_asr_result = []
second_stage_curve_asr_result = []

first_stage_asr_result = []
second_stage_asr_result = []

first_stage_acc_result = []
second_stage_acc_result = []


def logging_unified_results(args, ori_end_acc, ori_end_asr):
    dir_path = os.path.join(os.getcwd(), 'record/' + args.result_file + '/defense/mc_repair/results')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    import pickle
    filename = f"RandomSeed_{args.random_seed}_CurvePoint_{args.curve_t}_Ratio_{args.ratio}_FixStart_{args.fix_start}_FixEnd_{args.fix_end}_Epoch_{args.epochs}_PDMCEpoch_{args.PDMC_round}.pkl"
    file_path = os.path.join(dir_path, filename)

    with open(file_path, 'wb') as file:
        pickle.dump((ori_end_acc, ori_end_asr, first_stage_curve_acc_result, second_stage_curve_acc_result,
                     first_stage_curve_asr_result, second_stage_curve_asr_result,
                     first_stage_asr_result, second_stage_asr_result,
                     first_stage_acc_result, second_stage_acc_result), file)


def test_result(arg, testloader_cl, testloader_bd, epoch, model, criterion, device):
    model.eval()

    total_clean_test, total_clean_correct_test, test_loss = 0, 0, 0
    for i, (inputs, labels, *additional_info) in enumerate(testloader_cl):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        total_clean_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean_test += inputs.shape[0]
        avg_acc_clean = float(total_clean_correct_test.item() * 100.0 / total_clean_test)
    # progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
    print(
        'Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct_test, total_clean_test))
    logging.info(
        'Epoch:{} | Test Acc: {:.3f}%({}/{})'.format(epoch, avg_acc_clean, total_clean_correct_test, total_clean_test))

    total_backdoor_test, total_backdoor_correct_test, test_loss = 0, 0, 0
    for i, (inputs, labels, *additional_info) in enumerate(testloader_bd):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        total_backdoor_correct_test += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_backdoor_test += inputs.shape[0]
        avg_acc_bd = float(total_backdoor_correct_test.item() * 100.0 / total_backdoor_test)
    # progress_bar(i, len(testloader), 'Test %s ACC: %.3f%% (%d/%d)' % (word, avg_acc_clean, total_clean_correct, total_clean))
    print('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_bd, total_backdoor_correct_test,
                                                       total_backdoor_test))
    logging.info('Epoch:{} | Test Asr: {:.3f}%({}/{})'.format(epoch, avg_acc_bd, total_backdoor_correct_test,
                                                              total_backdoor_test))
    return avg_acc_bd, avg_acc_clean


class mc_repair_Class(defense):
    def __init__(self, args):
        # with open(args.yaml_path, 'r') as f:
        #     defaults = yaml.safe_load(f)
        with open(args.yaml_path, "r", encoding="utf-8-sig") as f:
            defaults = yaml.safe_load(f)

        defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__:
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'],
                            help="dataloader pin_memory")
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'],
                            help=".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', default=False, type=lambda x: str(x) in ['True', 'true', '1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='gtrsb')

        parser.add_argument('--epochs', type=int, help='the epochs for mode connectivity')
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float, default=0)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)

        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                            help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')

        parser.add_argument('--acc_ratio', type=float, help='the tolerance ration of the clean accuracy')
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--print_every', type=int, help='print results every few iterations')
        parser.add_argument('--nb_iter', type=int, help='the number of iterations for training')

        # Parameter for MC-repair
        parser.add_argument('--repair_alpha', type=float, default=0.5, help='fusion alpha in [0, 1]')
        parser.add_argument('--saving_curve', type=int, default=0)

        parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
                            help='curve type to use (default: None)')
        parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                            help='number of curve bends (default: 3)')

        parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                            help='fix start point (default: off)')
        parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                            help='fix end point (default: off)')

        parser.add_argument('--curve_t', type=float, default=0.4, help='middle point position')
        parser.add_argument('--regular', type=int, default=1, help='use curve regularization')

        parser.add_argument('--PDMC_round', type=int, default=3, help='epochs for PDMC')

        parser.set_defaults(init_linear=True)
        parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                            help='turns off linear initialization of intermediate points (default: on)')

        parser.add_argument('--repair_mode', type=str, default='mc-repair',
                            choices=['fusion-variance', 'variance', 'mc-repair'])
        parser.add_argument('--hessian_size', type=int, default=100, help='batch size for hessian evaluation')

        parser.add_argument('--model_path', type=str,
                            help='the path of model to prune, default saved in attack_result.pt')

        parser.add_argument('--num_testpoints', type=int, default=20)
        parser.add_argument('--sam_rho', type=float, default=0.05, help='SAM first step bound')

        parser.add_argument('--ft_lr', type=float, default=0.0001, help='fine-tuning learning rate for MCR')
        parser.add_argument('--ft_epochs', type=int, default=10, help='fine-tuning epochs for MCR')

        parser.add_argument('--ft_lr_scheduler', type=str, default='CosineAnnealingLR',
                            help='the scheduler of lr for fine tuning')

    def set_result(self, result_file):
        # attack_file = 'record/' + result_file
        # save_path = 'record/' + result_file + '/defense/mc_repair/'
        attack_file =  result_file
        save_path =  result_file + '/defense'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save)
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(
            args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')

    def set_devices(self):
        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in self.args.device[5:].split(',')][0]}" if "," in self.args.device else self.args.device
                # since DataParallel only allow .to("cuda")
            ) if torch.cuda.is_available() else "cpu"
        )

    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        args = self.args
        result = self.result

        train_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=True)
        clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        data_all_length = len(clean_dataset)
        ran_idx = choose_index(self.args, data_all_length)
        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)
        data_set_without_tran = clean_dataset
        data_set_clean = self.result['clean_train']
        data_set_clean.wrapped_dataset = data_set_without_tran
        data_set_clean.wrap_img_transform = train_tran
        # data_set_clean.wrapped_dataset.getitem_all = False

        data_loader = torch.utils.data.DataLoader(data_set_clean, batch_size=self.args.batch_size,
                                                  num_workers=args.num_workers, shuffle=True)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        # data_bd_testset.wrapped_dataset.getitem_all = False
        poison_test_loader = DataLoader(data_bd_testset,
                                        batch_size=32 if args.dataset == 'imagenet100' else args.batch_size,
                                        num_workers=args.num_workers,
                                        drop_last=False, shuffle=True, pin_memory=True)

        test_tran = get_transform(self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        clean_test_loader = DataLoader(data_clean_testset,
                                       batch_size=32 if args.dataset == 'imagenet100' else args.batch_size,
                                       num_workers=args.num_workers,
                                       drop_last=False, shuffle=True, pin_memory=True)

        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = clean_test_loader
        test_dataloader_dict["bd_test_dataloader"] = poison_test_loader

        model = generate_cls_model(args.model, args.num_classes)
        if 'model_path' in args:
            abs_model_path = os.getcwd() + args.save_path + '/' + args.model_path
            state_dict = torch.load(abs_model_path)
            logging.info(f'Loading model from {abs_model_path}')
        else:
            state_dict = self.result['model']

        # Prepare model, optimizer, scheduler
        model.load_state_dict(state_dict)
        # model.to(args.device)

        self.repair_train(args, model, data_loader, data_loader, poison_test_loader, clean_test_loader)

        result = {}
        # result['model'] = model_mcr
        # save_defense_result(
        #     model_name=args.model,
        #     num_classes=args.num_classes,
        #     model=model_mcr.cpu().state_dict(),
        #     save_path=args.save_path,
        # )
        return result

    def defense(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result

    def mode_connectivity_Point(self, args, model_a, model_b, data_train, bd_test, clean_test, PDMC_round, curve_name, point_t):
        architecture = get_curve_class(args)
        assert (args.curve is not None)
        curve = getattr(curve_models.curves, args.curve)
        # initialize curve
        model = curve_models.curves.CurveNet(
            args.num_classes,
            curve,
            architecture.curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=architecture.kwargs_curve,
        )
        if torch.cuda.is_available():
            model = model.cuda()

        # import end model
        model.import_base_parameters(model_a, 0)
        model.import_base_parameters(model_b, args.num_bends - 1)

        if args.init_linear:
            print('Linear initialization.')
            model.init_linear()

        if torch.cuda.device_count() > 1 and "," in args.device:
            logging.info(f"device={args.device}, trans curve model from cuda to DataParallel")
            save_model = deepcopy(model)
            save_model.cpu()
            model = torch.nn.DataParallel(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        criterion = nn.CrossEntropyLoss()
        regular = None
        if args.regular > 0:
            logging.info('using l2 regularization.......')
            regular = curve_models.curves.l2_regularizer(args.wd)

        logging.info(f'$$$$$$$$$$$$$$$$$ Train_Curves {curve_name} $$$$$$$$$$$$$$$$$')

        for epoch in range(args.epochs):
            # 在 mode_connectivity_Point 的 for epoch ... 循环里开头加入：
            args._epoch_idx = epoch
            batch_loss = oneEpochTrain(args, model, data_train, criterion, optimizer, scheduler, self.device, regular)
            logging.info(f'Train_Curves on Clean Data, epoch:{epoch} ,epoch_loss: {batch_loss}')

        if torch.cuda.device_count() > 1 and "," in args.device:
            logging.info("device='cuda', trans curve model from DataParallel to cuda")
            save_model.load_state_dict(model.module.cpu().state_dict())
            model = save_model

        single_device = torch.device('cuda') if torch.cuda.is_available() else "cpu"

        model.cpu()
        point_list = [point_t]
        point_model = []
        torch.cuda.empty_cache()
        time.sleep(5)

        for t in point_list:
            model_t = curve_opt.get_PointOnCurve_device(args=args, architecture=architecture,
                                                        curve_model=model, t=t, train_loader=data_train,
                                                        device=single_device)
            if torch.cuda.is_available():
                model_t = model_t.cuda()
            logging.info(f"Testing Curve {curve_name} t={t}")
            test_result(args, clean_test, bd_test, PDMC_round, model_t, nn.CrossEntropyLoss(), single_device)
            point_model.append(model_t.cpu())
            model_t = None
            torch.cuda.empty_cache()

        if curve_name is None:
            curve_path = os.path.join(os.getcwd(), args.checkpoint_save, f'curve_result-ratio{args.ratio}.pt')
        else:
            curve_path = os.path.join(os.getcwd(), args.checkpoint_save,
                                      f'curve_result_{curve_name}-ratio{args.ratio}.pt')

        torch.save(model.cpu().state_dict(), curve_path)

        adv_test_curve = curve_models.curves.CurveNet(
            args.num_classes,
            curve,
            architecture.curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=architecture.kwargs_curve,
        )
        adv_test_curve.load_state_dict(torch.load(curve_path))

        logging.info(f'Test_Curves on Adv_Set ')

        adv_result = testCurve(model=adv_test_curve, device=single_device, train_loader=data_train,
                               test_loader=bd_test,
                               args=args, regular=curve_models.curves.l2_regularizer(args.wd))
        for key, value in adv_result.items():
            logging.info(f'Test {key}: {value}')

        adv_test_curve = None
        time.sleep(5)
        torch.cuda.empty_cache()

        cl_test_curve = curve_models.curves.CurveNet(
            args.num_classes,
            curve,
            architecture.curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=architecture.kwargs_curve,
        )
        cl_test_curve.load_state_dict(torch.load(curve_path))

        logging.info(f'Test_Curves on CL_Set')
        cl_result = testCurve(model=cl_test_curve, device=single_device, train_loader=data_train,
                              test_loader=clean_test,
                              args=args, regular=curve_models.curves.l2_regularizer(args.wd))
        for key, value in cl_result.items():
            logging.info(f'Test {key}: {value}')

        cl_test_curve = None
        torch.cuda.empty_cache()

        if args.saving_curve == 0 and os.path.exists(curve_path):
            os.remove(curve_path)

        return point_model, adv_result, cl_result

    def mc_repair(self, args, model_a, model_b, clean_sub, bd_test, clean_test, PDMC_round, maximize=False, name_a='', name_b='',
                  one_layer_per=False):
        """
        Maximize = False: make permutation to un-align the model
        Maximize = True: make permutation to align the model
        """
        t0__mc = step_begin(f"mc_repair(maximize={maximize}, name_a={name_a}, name_b={name_b})")
        if torch.cuda.is_available():
            model_b = model_b.cuda()
            model_a = model_a.cuda()

        model_a.eval()
        model_b.eval()

        if args.model == 'resnet18':
            model_b = permutation.find_permutation_ResNet18(model_a, model_b, clean_sub, maximize)
        elif args.model == 'preactresnet18':
            if one_layer_per:
                model_b = permutation.find_permutation_PreAct_ResNet18_one_layer(model_a, model_b, clean_sub, maximize)
            else:
                model_b = permutation.find_permutation_PreAct_ResNet18(model_a, model_b, clean_sub, maximize)
        else:
            raise ValueError(f"NO MODEL: {args.model}")


        device = torch.device(
            (
                "cuda"
            ) if torch.cuda.is_available() else "cpu"
        )

        logging.info(f"Testing model_a {name_a}...")
        test_result(args, clean_test, bd_test, PDMC_round, model_a, nn.CrossEntropyLoss(), device)

        logging.info(f"Testing model_b {name_b} after Permutation...")
        test_result(args, clean_test, bd_test, PDMC_round, model_b, nn.CrossEntropyLoss(), device)


        step_end(f"mc_repair(maximize={maximize}, name_a={name_a}, name_b={name_b})", t0__mc)
        return model_b

    def repair_another(self, args, model, clean_train_loader, bd_test, clean_test):
        t = args.curve_t
        PDMC_round = args.PDMC_round

        ts = np.linspace(0.0, 1.0, args.num_testpoints + 1).tolist()
        t_index = ts.index(t)

        clean_train_loader_sbatch = torch.utils.data.DataLoader(clean_train_loader.dataset, batch_size=128,
                                                                num_workers=args.num_workers, shuffle=True)

        clean_train_loader_augment = clean_train_loader


        logging.info('Metrics for the Original Backdoored model')
        ori_end_asr, ori_end_acc = test_result(args, clean_test, bd_test, -1, model, nn.CrossEntropyLoss(), 'cuda')

        model_end = deepcopy(model)

        for i in range(PDMC_round):
            # --- STEP B: Symmetric Permutation (maximize=False) ---
            t0 = step_begin(f"B: round={i}")
            model_b = self.mc_repair(args, model, model_end,
                                     clean_train_loader_sbatch, bd_test, clean_test, PDMC_round,
                                     maximize=False, name_a=f'{i} round Ori_backdoor', name_b=f'{i} round Ori_backdoor',
                                     one_layer_per=False)

            dt_b, mem_b, fwd_b, bwd_b = step_end(f"B: round={i}", t0)
            asr_b, acc_b = test_result(args, clean_test, bd_test, PDMC_round,
                            model_b.cuda() if torch.cuda.is_available() else model_b,
                            nn.CrossEntropyLoss(), 'cuda' if torch.cuda.is_available() else 'cpu')
            ra_b = 100.0 - asr_b
            csv_write_row(self.args, i, "B", None, acc_b, asr_b, ra_b, acc_b, dt_b, mem_b, fwd_b, bwd_b)

            torch.save(model_b.state_dict(), os.path.join(self.args.checkpoint_save, f"stepB_round{i}.pt"))

            # --- STEP C: First Curve (gamma_1) ---
            t0 = step_begin(f"C: round={i}, curve_t={t}")

            self.args._round_idx = i
            self.args._phase_tag = "C"
            models_t1, adv_result1, cl_result1 = self.mode_connectivity_Point(
                args, model, model_b, clean_train_loader_augment, bd_test, clean_test, PDMC_round,
                curve_name=f'{i}_round_First-Stage-t={t}', point_t=t
            )

            model_c = models_t1[0]
            dt_c, mem_c, fwd_c, bwd_c = step_end(f"C: round={i}, curve_t={t}", t0)
            asr_c, acc_c = test_result(args, clean_test, bd_test, PDMC_round,
                            model_c.cuda() if torch.cuda.is_available() else model_c,
                            nn.CrossEntropyLoss(), 'cuda' if torch.cuda.is_available() else 'cpu')
            ra_c = 100.0 - asr_c
            csv_write_row(self.args, i, "C", None, acc_c, asr_c, ra_c, acc_c, dt_c, mem_c, fwd_c, bwd_c)
            torch.save(model_c.state_dict(), os.path.join(self.args.checkpoint_save, f"stepC_round{i}.pt"))

            torch.cuda.empty_cache()
            # --- STEP D: Output-Consistency Permutation (maximize=True) ---
            t0 = step_begin(f"D: round={i}")
            model_ori_per = self.mc_repair(
                args, models_t1[0], model, clean_train_loader_sbatch, bd_test, clean_test, PDMC_round,
                maximize=True, name_a=f"{i} round First-Stage-t={t}", name_b=f'{i} round Ori_backdoor',
                one_layer_per=False
            )

            dt_d, mem_d, fwd_d, bwd_d = step_end(f"D: round={i}", t0)
            asr_d, acc_d = test_result(args, clean_test, bd_test, PDMC_round,
                            model_ori_per.cuda() if torch.cuda.is_available() else model_ori_per,
                            nn.CrossEntropyLoss(), 'cuda' if torch.cuda.is_available() else 'cpu')
            ra_d = 100.0 - asr_d
            csv_write_row(self.args, i, "D", None, acc_d, asr_d, ra_d, acc_d, dt_d, mem_d, fwd_d, bwd_d)
            torch.save(model_ori_per.state_dict(), os.path.join(self.args.checkpoint_save, f"stepD_round{i}.pt"))

            # --- STEP E: Second Curve (gamma_2) ---
            t0 = step_begin(f"E: round={i}, curve_t={t}")
            self.args._round_idx = i
            self.args._phase_tag = "E"
            models_t2, adv_result2, cl_result2 = self.mode_connectivity_Point(
                args, model_ori_per, models_t1[0], clean_train_loader_augment, bd_test, clean_test, PDMC_round,
                curve_name=f'{i}_round_Second-Stage-t={t}', point_t=t
            )

            model_e = models_t2[0]
            dt_e, mem_e, fwd_e, bwd_e = step_end(f"E: round={i}, curve_t={t}", t0)
            asr_e, acc_e = test_result(args, clean_test, bd_test, PDMC_round,
                            model_e.cuda() if torch.cuda.is_available() else model_e,
                            nn.CrossEntropyLoss(), 'cuda' if torch.cuda.is_available() else 'cpu')
            ra_e = 100.0 - asr_e
            csv_write_row(self.args, i, "E", None, acc_e, asr_e, ra_e, acc_e, dt_e, mem_e, fwd_e, bwd_e)
            torch.save(model_e.state_dict(), os.path.join(self.args.checkpoint_save, f"stepE_round{i}.pt"))

            torch.cuda.empty_cache()

            model = models_t2[0]
            if i < (PDMC_round-1) and args.ft_epochs > 0:
                logging.info('fine-tuning end model...')
                optimizer_ft = torch.optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9, weight_decay=5e-4)
                if args.ft_lr_scheduler == 'ReduceLROnPlateau':
                    scheduler_ft = getattr(torch.optim.lr_scheduler, args.ft_lr_scheduler)(optimizer_ft)
                else:
                    scheduler_ft = getattr(torch.optim.lr_scheduler, args.ft_lr_scheduler)(optimizer_ft, T_max=args.ft_epochs)
                criterion_ft = nn.CrossEntropyLoss()
                self.args._round_idx = i
                self.args._phase_tag = "FT"
                for epoch in tqdm(range(args.ft_epochs)):
                    self.args._epoch_idx = epoch
                    model.to(self.device)
                    oneEpochTrain(args, model, clean_train_loader, criterion_ft, optimizer_ft, scheduler_ft, self.device, None)

            model_end = deepcopy(model)

            first_stage_curve_acc_result.append(cl_result1)
            second_stage_curve_acc_result.append(cl_result2)
            first_stage_acc_result.append(cl_result1['acc'][t_index])
            second_stage_acc_result.append(cl_result2['acc'][t_index])

            first_stage_curve_asr_result.append(adv_result1)
            second_stage_curve_asr_result.append(adv_result2)
            first_stage_asr_result.append(adv_result1['acc'][t_index])
            second_stage_asr_result.append(adv_result2['acc'][t_index])

        logging_unified_results(args, ori_end_acc, ori_end_asr)

    def repair_train(self, args, model, data_loader, clean_train_loader, bd_test, clean_test):
        # model_clean = deepcopy(model).cuda()
        if torch.cuda.is_available():
            model = model.cuda()

        if args.repair_mode == 'mc_repair':
            self.repair_another(args, model, clean_train_loader, bd_test, clean_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    mc_repair_Class.add_arguments(parser)
    args = parser.parse_args()
    mc_repair_method = mc_repair_Class(args)
    result = mc_repair_method.defense(args.result_file)
