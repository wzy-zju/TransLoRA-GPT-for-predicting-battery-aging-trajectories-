import numpy as np
import random
import torch
import torch.nn as nn
import os
import math
import glob
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from src.gpt2_utils.gpt2_book import LinearWithLoRA



def set_global_seed(seed):
    torch.manual_seed(seed)          # PyTorch
    torch.cuda.manual_seed_all(seed) # CUDA
    np.random.seed(seed)             # NumPy
    random.seed(seed)               # Python built-in random
    # Ensure deterministic convolution operations (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evalute_model(model, train_loader, val_loader, config):
    model.eval()
    with torch.no_grad():
        train_loss, train_loss_list = calc_loss_loader(train_loader, model, config["device"], config)
        val_loss, val_loss_list = calc_loss_loader(val_loader, model, config["device"], config)
    return train_loss, val_loss , train_loss_list , val_loss_list





def compute_policy_loss(y_pred, label_y, config):
    assert y_pred.shape == label_y.shape, (f"Prediction tensor and label tensor shapes must be consistent! ")
    if config['loss'] ==  "MSE": 
        loss_function = nn.MSELoss()
        loss = loss_function(y_pred, label_y)
    elif config['loss'] ==  "MAE": 
        loss_function = nn.L1Loss()
        loss = loss_function(y_pred, label_y)
    elif config['loss'] ==  "MAPE":
        epsilon = 1e-8
        loss = torch.mean(torch.abs((label_y - y_pred) / (label_y + epsilon)))

    return loss

def calc_loss_loader(data_loader, model, device, config):
    total_loss = 0
    num_batches = len(data_loader)

    if config['data_process']['one_step']:
        for i, (input_batch, target_batch) in enumerate(data_loader):
            input_batch, target_batch = input_batch.to(device).float(), target_batch.to(device).float()
            y_pred, _ = model(input_batch, None)
            loss = compute_policy_loss(y_pred, target_batch, config)
            total_loss += loss.item()
        return total_loss / num_batches , None
    else:
        for i, (x_uiui, y_uiui, x_uisoh, y_uisoh) in enumerate(data_loader):
            x_uiui = x_uiui.to(config["device"]).float()
            y_uiui = y_uiui.to(config["device"]).float()
            x_uisoh = x_uisoh.to(config["device"]).float()
            y_uisoh = y_uisoh.to(config["device"]).float()

            output_ui, output_soh = model(x_uiui, x_uisoh)

            loss_uiui = compute_policy_loss(output_ui, y_uiui, config) 
            loss_uisoh = compute_policy_loss(output_soh, y_uisoh, config)

            alpha = config['loss_alpha'] 
            total_loss += (alpha * loss_uiui + (1 - alpha) * loss_uisoh).item()
            loss_list = [loss_uiui, loss_uisoh]
        return total_loss / num_batches , loss_list



def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    trainable_percentage = 100 * trainable_params / total_params
    print(f"Trainable parameter percentage: {trainable_percentage:.2f}%")
    return total_params, trainable_params, trainable_percentage

def compute_test_loss(model, test_ui_windows, test_soh_windows, test_x_baseline, config, test_y_uiui_windows=None):
    model.eval()
    with torch.no_grad():
        device = config["device"]
        if config['data_process']['one_step']:
            input_batch =  torch.from_numpy(test_ui_windows).to(device).float()
            target_batch = torch.from_numpy(test_soh_windows).to(device).float()
            target_x_baseline = torch.from_numpy(test_x_baseline).to(device).float()
            y_pred, _= model(input_batch, None)
            y_pred = change_baseline(y_pred, target_x_baseline)
            loss = compute_policy_loss(y_pred, target_batch, config)
            return loss, y_pred, target_batch
        else:
            
            test_x_uiui_windows = torch.from_numpy(test_ui_windows).to(device).float()
            test_y_uisoh_windows =  torch.from_numpy(test_soh_windows).to(device).float().reshape(-1)
            test_y_uiui_windows = torch.from_numpy(test_y_uiui_windows).to(device).float()
            
            output_soh = 1
            soh_trajectory_pred = [ ]
            for i in range(1000):
                output_ui, output_soh = model(test_x_uiui_windows, None)
                new_soh = output_soh[-1].item()
                soh_trajectory_pred.append(new_soh)  # (1, 50, 1) becomes (50) in model
                if new_soh>0.8:
                    next_step_pred = output_ui[:, -1:, :, :] # (1, 50, 40) -> (1, 40)
                    test_x_uiui_windows = torch.cat((test_x_uiui_windows[:, 1:,: , :], next_step_pred), dim=1)
                else:
                    break
            return soh_trajectory_pred, test_y_uisoh_windows, test_y_uisoh_windows


def change_baseline(y_pred, target_x_baseline):
    offset = target_x_baseline - y_pred[:,0]
    y_pred = y_pred + offset.unsqueeze(1)
    return y_pred



def plot_test(y_pred, y_label, config, save_path, test_id, rmse):
    x_label = np.arange(len(y_label))  # x-axis range for y_label (0 to len(y_label)-1)
    x_pred = np.arange(len(y_pred))    # x-axis range for y_pred (0 to len(y_pred)-1)

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False # Used to display minus signs normally

    plt.figure(figsize=(12, 7))

    plt.plot(x_label, y_label, color='b', linestyle='-', label='Truth')
    plt.plot(x_pred, y_pred, color='r', linestyle='--', marker='o', markersize=3, label='Prediction')
    plt.title('SOH vs. Prediction', fontsize=16)
    plt.xlabel('Cycles', fontsize=12)
    plt.ylabel('SOH', fontsize=12)
    plt.legend()
    plt.grid(True)


    save_dir = save_path
    now = datetime.datetime.now()
    # file_name = f"{now.month:02d}-{now.day:02d}-{now.hour:02d}-{now.minute:02d}--{num}-" + config['GPT2']['fine_tune'] + ".png"
    file_name = f"test_{test_id}_{rmse}.png"
    
    full_path = os.path.join(save_dir, file_name)
    plt.savefig(full_path, dpi=300)
    
    print(f"Image saved to: {full_path}")
    plt.show()



def save_results_to_excel(y_pred, y_label, config, epoch,total_params, trainable_params, trainable_percentage, elapsed_time):
    """
    Calculate evaluation metrics and save test results to formatted Excel file.

    Args:
        y_pred (np.array): Model prediction result array.
        y_label (np.array): True label array.
        config (dict): Dictionary containing all configuration information.
    """
    print("\n--- Preparing to save test results to Excel... ---")

    # --- a. Parse current test battery information from config ---
    test_name = None
    test_id = None
    for dataset_info in config['data_process']['test_sets']:
        # Find the first test set containing files (i.e., non-empty)
        if dataset_info['files']:
            test_name = dataset_info['name']
            # Assume each test targets only one file
            test_id = dataset_info['files'][0]
            print(f"Detected test set: {test_name}, Battery ID: {test_id}")
            break

    if test_name is None:
        print("Warning: No test files specified in config test_sets, cannot save Excel results.")
        return

    # --- b. Calculate RMSE and MAPE metrics ---
    # Ensure y_pred and y_label are numpy arrays
    if not isinstance(y_pred, np.ndarray): y_pred = y_pred.numpy()
    if not isinstance(y_label, np.ndarray): y_label = y_label.numpy()
    
    rmse = np.sqrt(np.mean((y_pred - y_label) ** 2))
    # Add a small value to denominator to avoid division by zero if y_label contains 0
    mape = np.mean(np.abs((y_label - y_pred) / (y_label + 1e-8))) * 100
    
    print(f"Calculated metrics: RMSE = {rmse:.4f}, MAPE = {mape:.4f}%")
    
    # --- c. Prepare data to write to Excel ---
    # Create a dictionary with column names as keys and data as values. This handles cases where column lengths differ.
    epochs_per_second = (epoch + 1) / elapsed_time if elapsed_time > 0 else 0

    epoch_time_col_data = [
        f"Epochs trained: {epoch + 1}",
        f"Elapsed time (s): {elapsed_time:.2f}",
        f"Epochs per second: {epochs_per_second:.4f}"
    ]

    params_col_data = [
        f"Total params: {total_params}",
        f"Trainable params: {trainable_params}",
        f"Trainable percentage: {trainable_percentage:.2f}%"
    ]
    # 3. Create a dictionary with column names as keys and data as values.
    #    Use max_len to handle cases where column lengths differ, fill gaps with None.
    max_len = max(len(y_pred), len(y_label), len(epoch_time_col_data), len(params_col_data))
    
    data_dict = {
        'pred': list(y_pred) + [None] * (max_len - len(y_pred)),
        'label': list(y_label) + [None] * (max_len - len(y_label)),
        'RMSE': [f"{rmse:.6f}"] + [None] * (max_len - 1),  # Format metrics as strings too
        'MAPE (%)': [f"{mape:.4f}"] + [None] * (max_len - 1),
        # --- Two new columns ---
        'epoch_time': epoch_time_col_data + [None] * (max_len - len(epoch_time_col_data)),
        'params': params_col_data + [None] * (max_len - len(params_col_data))
    }
    
    df = pd.DataFrame(data_dict)


    # --- d. Build save path and save file ---
    # Create folder based on dataset name, e.g., CX2 -> CX
    folder_name = test_name[:2]
    save_path = os.path.join('save_result', folder_name)
    
    # Create folder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Build complete filename and save
    excel_filename = f"test_{test_id}_{rmse:.4f}.xlsx"
    full_path = os.path.join(save_path, excel_filename)
    
    try:
        df.to_excel(full_path, index=False, sheet_name='Sheet1')
        print(f"✅ Results successfully saved to: {full_path}")
    except Exception as e:
        print(f"Error: Failed to save Excel file. Error message: {e}")
    return save_path, test_id, rmse



def save_model(model, config, val_loss):
    if config['data_process']['one_step']:
        save_dir = 'save_model/one_step'
    else:
        save_dir = 'save_model/auto_regression'

    if  config['GPT2']['fine_tune']:
        search_pattern = os.path.join(save_dir, '*freeze_all_gpt.pth')
    else:
        search_pattern = os.path.join(save_dir, '*freeze_without_norm.pth')
    search_pattern = os.path.join(save_dir, '*' + config['GPT2']['fine_tune'] + '.pth')

    existing_models = glob.glob(search_pattern)

    best_loss = float('inf') # Initialize best loss to infinity
    best_model_path = None
    
    # Iterate through all models, select the best one
    if existing_models:
        for model_path in existing_models:
            loss_str = os.path.basename(model_path).split('_')[0]
            loss_val = float(loss_str)/1000
            # if loss_val < best_loss:
            best_loss = loss_val
            best_model_path = model_path

    

    print(f"loss={math.sqrt(val_loss):.2f}")
    if val_loss < best_loss:
        new_model_save_path = os.path.join(save_dir, f"{math.sqrt(val_loss):.2f}_" + config['GPT2']['fine_tune'] + ".pth")

        if  config['GPT2']['fine_tune'] == 'freeze_all_gpt':
            weights_to_save = {
                'in_layer_state_dict': model.in_layer.state_dict(),
                'out_layer_1_state_dict': model.out_layer_1.state_dict(),
                'out_layer_2_state_dict': model.out_layer_2.state_dict(),
                "global_mean":model.global_mean,
                "global_std":model.global_std
                }  
            torch.save(weights_to_save, new_model_save_path)
        
        if  config['GPT2']['fine_tune'] == 'no_freeze':
            torch.save(model.state_dict(), new_model_save_path)

        if  config['GPT2']['fine_tune'] == 'LoRA':
            torch.save(model.state_dict(), new_model_save_path)
        




def load_model(model, config, model_path=None):
    if config['data_process']['one_step']:
        save_dir = 'save_model/one_step'
    else:
        save_dir = 'save_model/auto_regression'
    
    if model_path is None:
        search_pattern = os.path.join(save_dir, '*' + config['GPT2']['fine_tune'] + '.pth')
        existing_models = glob.glob(search_pattern)

        if not existing_models:
            raise ValueError("Warning: No saved model found in directory. Using initialized weights.")

        best_loss = float('inf')
        best_model_path = None
        
        for path in existing_models:
            loss_str = os.path.basename(path).split('_')[0]
            loss_val = float(loss_str) / 1000.0
            
            if loss_val < best_loss:
                best_loss = loss_val
                best_model_path = path

        model_path_to_load = best_model_path
    else:
        model_path_to_load = model_path

    if  config['GPT2']['fine_tune'] == 'freeze_all_gpt':
        assert model_path_to_load is not None, "model_path_to_load is None, please check model loading part!"
        checkpoint = torch.load(model_path_to_load)
        model.in_layer.load_state_dict(checkpoint['in_layer_state_dict'])
        model.out_layer_1.load_state_dict(checkpoint['out_layer_1_state_dict'])
        model.out_layer_2.load_state_dict(checkpoint['out_layer_2_state_dict'])
        model.global_mean = checkpoint['global_mean']
        model.global_std = checkpoint['global_std']
        if 'gpt2_finetuned_state_dict' in checkpoint:
            gpt2_state_dict = model.gpt2.state_dict()  
            gpt2_state_dict.update(checkpoint['gpt2_finetuned_state_dict'])
            model.gpt2.load_state_dict(gpt2_state_dict)
    else:
        assert model_path_to_load is not None, "model_path_to_load is None, please check model loading part!"
        state_dict_to_load = torch.load(model_path_to_load)
        model.load_state_dict(state_dict_to_load)

    model.eval() # After loading model, usually switch to evaluation mode
    return model


def smooth_pred(y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.numpy()

    s = pd.Series(y_pred)
    window_size=10  # Larger value means smoother
    rolling_mean = s.rolling(window=window_size, min_periods=1, center=True).mean()
    
    # Convert result back to NumPy array
    assert isinstance(rolling_mean, pd.Series), "rolling_mean is not pd.Series type"
    return rolling_mean.to_numpy()



def gpt_requires_grad(config, gpt2):
    if config['GPT2']['fine_tune'] == 'freeze_all_gpt':    
        for i, (name, param) in enumerate(gpt2.named_parameters()):
            param.requires_grad = False
    elif config['GPT2']['fine_tune'] == 'no_freeze':
        for i, (name, param) in enumerate(gpt2.named_parameters()):
            param.requires_grad = True
    elif config['GPT2']['fine_tune'] == 'LoRA':
        for i, (name, param) in enumerate(gpt2.named_parameters()):
            # if "pos_emb" in name:
                # param.requires_grad = True
            # else:
            param.requires_grad = False
        replace_linear_with_lora(gpt2, rank=config['GPT2']['rank'], alpha=config['GPT2']['alpha'])



        # for i, (name, param) in enumerate(gpt2.named_parameters()):
        #     if "norm" in name or "pos_emb" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False


def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha)