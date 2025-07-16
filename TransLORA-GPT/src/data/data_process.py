from torch.utils.data import Dataset
import torch
import numpy as np
import os
import sys
os.chdir(sys.path[0])


class UI_Dataset(Dataset):
    def __init__(self, data, config):
        self.config = config
        if self.config['data_process']['one_step']:
            self.x_uiui_data = data[0]
            self.y_soh_data = data[1]
        else:
            self.x_uiui_data = data[0]
            self.y_uiui_data = data[1]
            self.y_soh_data = data[2]

    def __len__(self):
        return len(self.x_uiui_data)

    def __getitem__(self, idx):
        if self.config['data_process']['one_step']:
            one_x = self.x_uiui_data[idx]
            one_y = self.y_soh_data[idx]
            return one_x, one_y
        else:
            x_uiui = self.x_uiui_data[idx]
            y_uiui = self.y_uiui_data[idx].copy()
            x_uisoh = self.y_uiui_data[idx].copy()
            y_uisoh = self.y_soh_data[idx]
            return x_uiui, y_uiui, x_uisoh, y_uisoh


def cap_to_soh(capacity_data, config):
    rated_cap = capacity_data[0]
    soh_data  = capacity_data / rated_cap

    if config['data_process']['80%_truncate']:
        eol_index = np.argmin(np.abs(soh_data - 0.8))
        truncation_idx = eol_index + 1
        return soh_data[:truncation_idx], truncation_idx
    else:
        return soh_data, None



    # eol_index = np.argmin(np.abs(soh_data - 0.8))
    # truncation_idx = eol_index + 1
    return soh_data


def range_ui(ui_data, soc_range):
    # --- 1. Define original data SOC coordinate axis ---
    SOC_START = 0.1
    SOC_END = 0.73 
    num_points = ui_data.shape[-1]
    
    # (0.73 - 0.1) / (64 - 1) = 0.63 / 63 = 0.01
    soc_step = (SOC_END - SOC_START) / (num_points - 1)
    start_soc, end_soc = soc_range
    
    if start_soc >= end_soc:
        raise ValueError(f"SOC range start point {start_soc} must be less than end point {end_soc}.")
    if start_soc < SOC_START:
        raise ValueError(f"SOC range start point {start_soc} cannot be less than minimum SOC {SOC_START}.")
    if end_soc > SOC_END:
        raise ValueError(f"SOC range end point {end_soc} cannot be greater than maximum SOC {SOC_END}.")

    # --- 3. Convert SOC range to integer indices of array ---
    start_index = int(round((start_soc - SOC_START) / soc_step))
    end_index = int(round((end_soc - SOC_START) / soc_step))

    # --- 4. Use calculated indices to slice data ---
    sliced_ui_data = ui_data[..., start_index : end_index]

    return sliced_ui_data


def window_data_one_step(x, y, window_size, pred_len):
    # One-time prediction corresponding window cutting, i.e., x moves one step each time, and y also moves 1 step starting from window+pred_len
    num_samples = x.shape[0]
    
    
    num_windows = num_samples - window_size - pred_len + 1 # Already calculated

    assert num_samples == y.shape[0], f"Feature x and label y sample counts must be consistent! "
    assert num_windows > 0, f"pred_len is too long, can't split any windows!!"
    
    x_windows = []
    y_windows = []
    x_baseline = []
        
    for i in range(num_windows):
        x_start_index = i
        x_end_index = i + window_size
        
        y_start_index = x_end_index
        y_end_index = y_start_index + pred_len
        
        x_win = x[x_start_index:x_end_index]
        y_win = y[y_start_index:y_end_index]
        x_baseline_win = y[x_end_index-1]

        x_windows.append(x_win)
        y_windows.append(y_win)
        x_baseline.append(x_baseline_win)

    return np.array(x_windows), np.array(y_windows), np.array(x_baseline)



def window_data_autoregression(x, y, window_size):
    # Autoregressive mode, i.e.

    num_samples = x.shape[0]
    
    assert num_samples == y.shape[0], f"Feature x and label y sample counts must be consistent! "
    assert window_size <= num_samples, f"Window size {window_size} cannot exceed total sample count {num_samples}!"

    x_windows = []
    y1_windows = [] # UI window shifted by one position
    y2_windows = [] # SOH window shifted by one position
    
    # --- 2. Split all complete, non-overlapping windows ---
    # Set step size of range to window_size to achieve non-overlapping

    for i in range(0, num_samples, window_size):
        end_index = i + window_size
        
        if end_index+1 < num_samples:
            x_window = x[i:end_index]
            y1_window = x[i+1:end_index+1]
            y2_window = y[i+1:end_index+1]
            
            x_windows.append(x_window)
            y1_windows.append(y1_window)
            y2_windows.append(y2_window)

        # else:
        #     last_x_window = x[-window_size-1:-1]
        #     last_y1_window = x[-window_size:]
        #     last_y2_window = y[-window_size:]
            
        #     x_windows.append(last_x_window)
        #     y1_windows.append(last_y1_window)
        #     y2_windows.append(last_y2_window)

        #     break # Finished running, exit loop

    return np.array(x_windows), np.array(y1_windows), np.array(y2_windows)


def load_split_dataloader(config):
    # --- Step 1: Uniformly load all data defined in train_sets ---
    all_x_ui_data = []
    all_y_ui_data = []
    all_y_soh_data = []

    # Get data source definitions
    available_datasets = config['data_process']['datasets']
    for dataset_info in config['data_process']['train_sets']:
        dataset_name = dataset_info['name']
        dataset_files = dataset_info['files']
        base_ui = available_datasets[dataset_name]['base_ui_path']
        base_cap = available_datasets[dataset_name]['base_cap_path']
        for file_id in dataset_files:
            x, y1, y2, x_baseline = process_data(config, base_ui, base_cap, file_id)
            all_x_ui_data.append(x)
            all_y_ui_data.append(y1)
            all_y_soh_data.append(y2)


    # --- Step 2: Uniformly load all data defined in test_sets ---
    all_test_x = []
    all_test_y1 = []
    all_test_y2 = []
    all_test_x_baseline = []

    for dataset_info in config['data_process']['test_sets']:
        dataset_name = dataset_info['name']
        dataset_files = dataset_info['files']
        
        base_ui = available_datasets[dataset_name]['base_ui_path']
        base_cap = available_datasets[dataset_name]['base_cap_path']

        for file_id in dataset_files:
            x, y1, y2, x_baseline = process_data(config, base_ui, base_cap, file_id)
            all_test_x.append(np.expand_dims(x[0], axis=0))
            all_test_y1.append(np.expand_dims(y1[0], axis=0))
            all_test_y2.append(np.expand_dims(y2[0], axis=0))
            all_test_x_baseline.append(np.expand_dims(x_baseline[0], axis=0))


    x_ui_all = np.concatenate(all_x_ui_data, axis=0)
    y_ui_all = np.concatenate(all_y_ui_data, axis=0)
    y_soh_all = np.concatenate(all_y_soh_data, axis=0)
    

    # Merge all test data
    test_x_ui_windows = np.concatenate(all_test_x, axis=0)
    test_y_ui_windows = np.concatenate(all_test_y1, axis=0)
    test_y_soh_windows = np.concatenate(all_test_y2, axis=0)
    test_x_baseline = np.concatenate(all_test_x_baseline, axis=0)

    # Calculate global statistics (only on training data)
    global_mean = x_ui_all.mean(axis=(0, 1, 3))
    global_std = x_ui_all.std(axis=(0, 1, 3))
    #Note that this global_mean has 2 values! Respectively voltage and current values
    assert x_ui_all.shape[0] == y_soh_all.shape[0], f"Merged X and y sample counts don't match!"

    # --- Step 3: Split dataset ---
    num_samples = x_ui_all.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    x_ui_shuffled = x_ui_all[indices]
    y_ui_shuffled = y_ui_all[indices]
    y_soh_shuffled = y_soh_all[indices]

    train_ratio = config['data_process']['split_ratios']
    train_end_idx = int(num_samples * train_ratio)

    x_ui_train = x_ui_shuffled[:train_end_idx]
    y_ui_train = y_ui_shuffled[:train_end_idx]
    y_soh_train = y_soh_shuffled[:train_end_idx]

    x_ui_val = x_ui_shuffled[train_end_idx:]
    y_ui_val = y_ui_shuffled[train_end_idx:]
    y_soh_val = y_soh_shuffled[train_end_idx:]

    if config['data_process']['one_step']:# One-step testing. Use [0:50] ui to test SOH of next 100 cycles
        train_dataset = UI_Dataset((x_ui_train, y_soh_train), config)
        val_dataset = UI_Dataset((x_ui_val, y_soh_val),config)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['bsz'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['bsz'], shuffle=False)

        data_pack = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_x_ui_windows': test_x_ui_windows,
            'test_y_soh_windows': test_y_soh_windows,
            'test_x_baseline':test_x_baseline,
            "global_mean": global_mean,
            "global_std": global_std
        }
    else: # Can be tested in cycles.
        train_dataset = UI_Dataset((x_ui_train, y_ui_train, y_soh_train), config)
        val_dataset = UI_Dataset((x_ui_val, y_ui_val, y_soh_val), config)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['bsz'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['bsz'], shuffle=False)

        data_pack = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_x_ui_windows': test_x_ui_windows,
            'test_y_ui_windows': test_y_ui_windows,
            'test_y_soh_windows': test_y_soh_windows,
            "global_mean": global_mean,
            "global_std": global_std

        }


    # If it's one_step, then y_uiui has no meaning; it's directly x and soh
    return data_pack


    

def process_data(config, base_name_ui, base_name_cap, i):
    ui_file = base_name_ui.format(i)
    cap_file = base_name_cap.format(i)

    if not os.path.exists(ui_file) or not os.path.exists(cap_file):
        raise ValueError("File does not exist!")
        
    ui_data = np.load(ui_file)
    cap_data = np.load(cap_file)
        
    # Convert to SOH
    if config['data_process']['cap_to_soh']:
        cap_data, truncation_idx = cap_to_soh(cap_data, config)
    if truncation_idx is not None:
        ui_data = ui_data[:truncation_idx, :, :]

    ui_data = range_ui(ui_data, config['data_process']['soc_range'])

    # Split windows
    if config['data_process']['one_step']:
        x_ui_windows, y_soh_windows, x_soh_baseline = window_data_one_step(ui_data, cap_data, config['data_process']['window_size'], config['pred_len'] )
        y_ui_windows = np.zeros_like(y_soh_windows)
    else:
        x_ui_windows, y_ui_windows, y_soh_windows = window_data_autoregression(ui_data, cap_data, config['data_process']['window_size'])



    if x_ui_windows.shape[0] != y_soh_windows.shape[0]:
        raise ValueError("Two file dimensions don't match")
    
    return x_ui_windows, y_ui_windows , y_soh_windows, x_soh_baseline

