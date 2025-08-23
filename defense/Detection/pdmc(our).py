import torch
from ultralytics import YOLO
from collections import OrderedDict
import os
import copy  # For deepcopy if there's a need, though re-instantiation is preferred
import argparse
import yaml
from dataset.ultralytics_Dataset import PreloadedUltralyticsYOLODataset, DynamicUltralyticsYOLODataset, yolo_collate_fn
from torch.utils.data import DataLoader
from permutation_utils.yolov8_permutation import *
from copy import deepcopy
import time
import csv

def eval_asr(args, model_to_evaluate, experiment_name):
    results = model_to_evaluate.val(
        data=args.bd_test_data,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=experiment_name,
        device='cuda' if torch.cuda.is_available() else None,
        exist_ok=True,
    )
    print("\n--- Evaluation Summary ---")
    print(f"Results, detailed metrics, and plots are saved in: {results.save_dir}")

    # Print standard metrics
    print(f"  mAP@50-95 (All Classes): {results.box.map:.4f}")
    print(f"  mAP@50 (All Classes):    {results.box.map50:.4f}")
    print(f"  mAP@75 (All Classes):    {results.box.map75:.4f}")  # Added for more detail
    print(
        f"  Precision (All Classes): {results.box.p[0]:.4f}")  # Example: P for the first class or overall P if available differently
    print(f"  Recall (All Classes):    {results.box.r[0]:.4f}")  # Example: R for the first class or overall R

    header = [
        "mAP@50-95",
        "mAP@50",
        "mAP@75",
        "Precision",
        "Recall"
    ]

    # Define the data row (the corresponding values)
    data_row = [
        f"{results.box.map:.4f}",
        f"{results.box.map50:.4f}",
        f"{results.box.map75:.4f}",
        f"{results.box.p[0]:.4f}",
        f"{results.box.r[0]:.4f}"
    ]

    # Define the CSV filename
    csv_filename = f"{results.save_dir}/evaluation_results_poison.csv"

    # Writing to csv file
    with open(csv_filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the header
        csvwriter.writerow(header)
        # writing the data row
        csvwriter.writerow(data_row)

    # Calculate and print ASR
    # First, get class names from the dataset YAML used for this evaluation
    class_names = []
    try:
        with open(args.bd_test_data, 'r') as f:
            data_yaml_content = yaml.safe_load(f)
            class_names = data_yaml_content.get('names', [])
            if not isinstance(class_names, list) and isinstance(data_yaml_content.get('names'),
                                                                dict):  # handles {0: name}
                # Ensure sorted by key if it's a dict, for consistent indexing
                class_names_dict = data_yaml_content.get('names')
                class_names = [class_names_dict[i] for i in sorted(class_names_dict.keys())]

    except Exception as e:
        print(f"Warning: Could not read class names from {args.data} for ASR calculation: {e}")

    if args.asr_target_class in class_names:
        target_class_index = class_names.index(args.asr_target_class)

        # results.box.r is an array of recall values, one for each class, in order.
        if target_class_index < len(results.box.r):
            recall_target_class = results.box.r[target_class_index]
            asr = 1.0 - recall_target_class
            # print(f"\n--- Attack Success Rate (ASR) for class '{args.asr_target_class}' ---")
            print(f"  Recall for '{args.asr_target_class}' on this dataset: {recall_target_class:.4f}")
            # print(f"  ASR (1 - Recall for '{args.asr_target_class}'): {asr:.4f}")
        else:
            print(
                f"Warning: Index for '{args.asr_target_class}' out of bounds for recall metrics. Cannot calculate ASR.")
    else:
        print(
            f"Warning: Target class '{args.asr_target_class}' for ASR not found in dataset names: {class_names}. Cannot calculate ASR.")
        if not class_names:
            print(f"         (Class names list was empty or could not be loaded from {args.data})")


def extract_model_from_bezier_obj_input(
        model_A: YOLO,
        model_B: YOLO,
        t: float,
) -> YOLO:
    if not (0 <= t <= 1):
        raise ValueError("t must be in [0,1]")

    current_device = model_A.device if hasattr(model_A, 'device') and model_A.device is not None else \
        (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    weights_A = model_A.model.state_dict()
    weights_B = model_B.model.state_dict()

    interpolated_weights = OrderedDict()
    for key in weights_A:
        tensor_A = weights_A[key].float().to(current_device)
        tensor_B = weights_B[key].float().to(current_device)

        interpolated_tensor = (1 - t) * tensor_A + t * tensor_B
        interpolated_weights[key] = interpolated_tensor.cpu()  # 移回CPU，标准做法，加载时会自动移到模型设备

    interpolated_model_instance = YOLO(model=model_A.model_name, task=model_A.task)

    interpolated_model_instance.model.load_state_dict(interpolated_weights)
    interpolated_model_instance.to(current_device)  # 确保新模型也在正确的设备上

    return interpolated_model_instance


# fintune the extracted model after training
def finetune_bezier_interpolated_model_obj_input(
        args,
        model_A: YOLO,
        model_B: YOLO,
        t: float,
        experiment_name,
        data_yaml_path: str,
        **kwargs
):
    # extract model
    interpolated_model = extract_model_from_bezier_obj_input(
        model_A,
        model_B,
        t
    )
    print(f'The interpolated_model is index at {t}')
    print(f'Start fine-tuning the interpolated model.........')

    results = interpolated_model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        device='cuda' if torch.cuda.is_available() else None,
        name=experiment_name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        exist_ok=True,
        **kwargs
    )
    print(f"**********************************")
    print(f"**********************************")
    print(f"Reloading the last fine-tuned model: {args.project}/{experiment_name}/weights/last.pt")
    final_finetuned_model = YOLO(f"{args.project}/{experiment_name}/weights/last.pt")
    # Ensure the loaded model is on the same device as the original training was intended for
    # or a globally defined device for consistency.
    final_finetuned_model.to(interpolated_model.device)

    return final_finetuned_model


def permutation(model_A, model_B, maximize, data_loader, device):
    model_A.model.to(device)
    model_B.model.to(device)
    backbone_module_A = YOLOv8sBackbone()
    backbone_module_B = YOLOv8sBackbone()
    loadTorchYolov8(backbone_module_A, model_A)
    loadTorchYolov8(backbone_module_B, model_B)
    backbone_module_A.eval()  # freeze BN layer
    backbone_module_B.eval()
    print('Start permutation......')
    permutation_B = find_permutation_Yolov8_backbone(backbone_module_A.to(device), backbone_module_B.to(device),
                                                     data_loader, maximize=maximize)
    loadUltralyticsYolov8(permutation_B, model_B)
    backbone_module_A.cpu()
    backbone_module_B.cpu()
    torch.cuda.empty_cache()
    return model_B


def pdmc_defense(args):
    model_A = YOLO(args.weights)
    clean_dataset = DynamicUltralyticsYOLODataset(
        yaml_file=args.finetune_data, split='train', img_size=640, verbose=True
    )
    clean_loader = DataLoader(clean_dataset, batch_size=96, shuffle=True, collate_fn=yolo_collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    for i in range(args.pdmc_epochs):
        model_B = deepcopy(model_A)

        print(f"**********************************")
        print(f"**********************************")
        print(f'First stage, epochs {i}...............................')

        # === [Step 1] Symmetric permutation on backdoored models (PDMC Stage-B)
        # Goal: apply layer-wise neuron/channel permutation to break backdoor-aligned structure.
        # Implementation: maximize clean-loss proxy (here maximize=False if your permutation()
        #                 is coded to "find the most misaligned mapping" under this flag).
        # Outcome: obtain permuted model_B used as the endpoint for the first connectivity step.
        model_B = permutation(model_A, model_B, maximize=False, data_loader=clean_loader, device=device)

        # === [Step 2] Symmetric permutation–driven mode connectivity (PDMC Stage-C)
        # Goal: train a low-loss Bézier curve γ1 between (model_A, model_B); minimize E_t[L(γ1(t); D_clean)].
        # Outcome: curve-trained intermediate model; we then evaluate ASR and restore the model from checkpoint.
        model_tmp = finetune_bezier_interpolated_model_obj_input(
            args, model_A, model_B, args.t, f'first_stage_{i}', args.finetune_data
        )
        eval_asr(args, model_tmp, f'first_stage_asr_{i}')
        model_tmp = YOLO(f"{args.project}/first_stage_{i}/weights/last.pt")
        model_tmp.to(model_A.device)

        print(f"**********************************")
        print(f"**********************************")
        print(f'Second stage, epochs {i}...............................')

        # === [Step 3] Output-consistent permutation (PDMC Stage-D)
        # Goal: align the curve-selected/intermediate model (model_tmp) back to the clean-loss basin of model_A
        #       via output/feature-consistent assignment; implemented as a "maximize=True" variant in your API.
        # Outcome: get model_A_per that is permutation-aligned to stabilize the next connectivity.
        model_A_per = permutation(model_tmp, model_A, maximize=True, data_loader=clean_loader, device=device)

        # === [Step 4] Consistency permutation–driven connectivity (PDMC Stage-E)
        # Goal: train a second Bézier curve γ2 between (model_A_per, model_tmp) under clean-loss expectation,
        #       yielding a purified model while preserving clean performance.
        # Outcome: final purified model for this round; evaluate ASR and checkpoint it as new model_A.
        model_tmp = finetune_bezier_interpolated_model_obj_input(
            args, model_A_per, model_tmp, args.t, f'second_stage_{i}', args.finetune_data
        )
        eval_asr(args, model_tmp, f'second_stage_asr_{i}')
        model_tmp = YOLO(f"{args.project}/second_stage_{i}/weights/last.pt")
        model_tmp.to(model_A.device)

        # Update for next round
        model_A = model_tmp
        torch.cuda.empty_cache()
        time.sleep(5)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a YOLOv5/YOLOv8 model using Ultralytics.")
    parser.add_argument('--bd_test_data', type=str, default='/home/poisoned_coco_dis_subset/coco_subset.yaml',
                        help="Path to the poisoned test dataset YAML file")
    parser.add_argument('--finetune_data', type=str, default='/home/coco_yolo_finetune_5percent/coco_subset.yaml',
                        help="Path to the dataset YAML file (e.g., '/path/to/coco_subset.yaml').")
    parser.add_argument('--weights', type=str, default='/mnt/cocoDet/ckpts/yolov5s.pt',
                        help="'yolov5s.pt', 'yolov5m.pt', 'yolov5s.yaml'")
    parser.add_argument('--asr_target_class', type=str, default='person',
                        help="The target class name for ASR calculation (default: 'person').")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of training (fine-tuning) epochs.")
    parser.add_argument('--pdmc_epochs', type=int, default=3,
                        help="Number of pdmc epochs.")
    parser.add_argument('--batch', type=int, default=96,
                        help="Batch size for training. Use -1 for auto-batch (experimental).")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="Input image size (square, e.g., 640 for 640x640).")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to run on, e.g., 'cpu', '0' (for CUDA device 0), '0,1,2,3'. If None, auto-selects GPU if available.")
    parser.add_argument('--project', type=str, required=True,
                        help="Directory to save training runs.")
    parser.add_argument('--patience', type=int, default=20,
                        help="Epochs to wait for no observable improvement for early stopping.")
    parser.add_argument('--workers', type=int, default=8,
                        help="Number of worker threads for data loading (per RANK if DDP).")
    parser.add_argument('--optimizer', type=str, default='Adam',
                        # Ultralytics default is 'auto' which resolves to SGD or AdamW
                        help="Optimizer to use, e.g., 'SGD', 'Adam', 'AdamW'. 'auto' lets Ultralytics choose.")
    parser.add_argument('--lr0', type=float, default=0.005,  # Ultralytics default is 0.01
                        help="Initial learning rate (e.g., 0.01). If None, uses Ultralytics default.")

    parser.add_argument('--t', type=float, default=0.4,
                        help="curve index for pdmc, ranged in [0,1]")

    args = parser.parse_args()
    pdmc_defense(args)
