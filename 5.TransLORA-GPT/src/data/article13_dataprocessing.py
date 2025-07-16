import scipy.io
import numpy as np
import os
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
#Above are all system libraries
#os.chdir(sys.path[0])

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")

def read_R_data():
    for i in range(1, 9):
        # Construct filename
        file_name = f'autodl-tmp/INR18650_{i}__0.xls'
        
        # Read 'cycle' sheet from Excel file
        df = pd.read_excel(file_name, sheet_name='cycle')
        
        # Read data from 'Charge IR(Ω)' column, removing first row
        charge_ir_data = df['Charge IR(Ω)'].iloc[1:].values
        
        # Convert data to NumPy array
        charge_ir_np = np.array(charge_ir_data)
        
        # Save as .npy file
        np.save(f'INR18650_{i}_R.npy', charge_ir_np)
    return None
        


def read_voltage_data(Battery_list, config):
    # Assume config is a global variable or imported from elsewhere
    # For code completeness, use example configuration values here, you can modify according to actual situation
    num_sample = config['dataset']['num_sample']  # How many samples to extend backward from min_SOH
    min_soc = config['dataset']['min_soc']  # For example, if you want samples from [10%---50%], 0.1 is min_soc
    min_SOH = config['dataset']['min_SOH']  # Only cycles greater than this value will be processed to get ui data
    rated_capacity = config['dataset']['rated_capacity']  # Rated capacity

    all_data = []
    big_df = pd.DataFrame()  # Large DataFrame to store all data
    for battery in Battery_list:
        battery_number = battery.split('_')[1]
        file_prefix = f'INR18650_{battery_number}'
        xls_folder = 'autodl-tmp'
        # Filter files matching the prefix
        matching_files = [f for f in os.listdir(xls_folder) if f.startswith(f'{file_prefix}__')]
        matching_files.sort(key=lambda x: int(x.split('__')[1].split('.')[0]))
        for file_name in matching_files:#6 excel files   
            file_path = os.path.join(xls_folder, file_name)
            if not os.path.exists(file_path):
                continue
            # Read all worksheets from Excel file into a dictionary, keys are worksheet names, values are DataFrames
            xls_dict = pd.read_excel(file_path, sheet_name=None)
            xls_dict = {sheet_name: df for sheet_name, df in xls_dict.items() if sheet_name.startswith('record_')}
            # Merge all worksheet data into one DataFrame
            df = pd.concat(xls_dict.values(), ignore_index=True)
            big_df = pd.concat([big_df, df], ignore_index=True)#Put everything into one!! big_df!


    # Filter rows that meet the conditions
    filtered_df = big_df[(big_df['Cycle ID'] >= 2) & (big_df['Step Name'] == 'CCCV_Chg')]# Look at the id column here
    # Calculate SOC
    #filtered_df['SOC'] = filtered_df['Capacity(mAh)'] / rated_capacity
    filtered_df.loc[:, 'SOC'] = filtered_df['Capacity(mAh)'] / rated_capacity
    target_soc_list = []
    for i in range(num_sample):
        soc = min_soc + i * 0.01
        target_soc_list.append(soc)
    # Find U values corresponding to each soc in target_soc_list that are closest
    all_cycle_data = []
    unique_cycle_ids = filtered_df['Cycle ID'].unique()  # Get all unique Cycle IDs
    for cycle_id in unique_cycle_ids:
        cycle_df = filtered_df[filtered_df['Cycle ID'] == cycle_id]  # Filter data belonging to this Cycle ID
        i_list = []
        u_list = []
        for soc in target_soc_list:
            soc_diffs = np.abs(cycle_df['SOC'] - soc)
            closest_index = soc_diffs.index[soc_diffs == np.min(soc_diffs)][0]
            i_list.append(cycle_df.loc[closest_index, 'Current(mA)'])
            u_list.append(cycle_df.loc[closest_index, 'Voltage(V)'])
        i_tensor = torch.tensor(i_list)
        u_tensor = torch.tensor(u_list)
        cycle_data = torch.stack((i_tensor, u_tensor))
        all_cycle_data.append(cycle_data)

    all_data = torch.stack(all_cycle_data)#Excel 1 shape looks fine, excluding cycle 1, it's (300,2,64)
    saved = all_data.numpy()
    battery_number = Battery_list[0].split('_')[1]
    np.save(f'INR18650_{battery_number}_ui.npy', saved)
    return None

def read_capacity_data(Battery_list):
    # Assume config is a global variable or imported from elsewhere
    # For code completeness, use example configuration values here, you can modify according to actual
    for battery in Battery_list:
        battery_number = battery.split('_')[1]
        file_prefix = f'INR18650_{battery_number}'
        xls_folder = 'autodl-tmp'
        # Filter files matching the prefix
        matching_files = [f for f in os.listdir(xls_folder) if f.startswith(f'{file_prefix}__')]
        matching_files.sort(key=lambda x: int(x.split('__')[1].split('.')[0]))
        file_name = matching_files[0]#Take first file, i.e., INR18650_?__0
        file_path = os.path.join(xls_folder, file_name)
        if not os.path.exists(file_path):
            continue
        df = pd.read_excel(file_path, sheet_name='cycle')
        # Filter data with Cycle ID >= 2
        filtered_df = df[df['Cycle ID'] >= 2]
        # Extract Cycle ID and Cap_Chg(mAh) column data
        result_df = filtered_df[ 'Cap_Chg(mAh)']
        saved = result_df.to_numpy()
        np.save(f'INR18650_{battery_number}_cap.npy', saved)
        return None


def load_one_data(config):
    loaded_dict = np.load(config['dataset']['path'], allow_pickle=True)
    return loaded_dict




def load_all_data():
    all_battery_capacity_list = []
    all_battery_ui_list = []
    for x in range(1, 9):  # Assume x ranges from 1 to 8
        # Load INR18650_x_cap.npy file
        cap_file_path = f'INR18650_{x}_cap.npy'
        try:
            cap_data = np.load(cap_file_path)
            all_battery_capacity_list.append(cap_data)
        except FileNotFoundError:
            print(f"File {cap_file_path} not found")
        # Load INR18650_x_ui.npy file
        ui_file_path = f'INR18650_{x}_ui.npy'
        try:
            ui_data = np.load(ui_file_path)
            all_battery_ui_list.append(ui_data)
        except FileNotFoundError:
            print(f"File {ui_file_path} not found")
    return all_battery_capacity_list, all_battery_ui_list



def change_ui(): #Convert input ui data to ensure (cycle,2,64) is voltage first then current, and current is in A units!

    # Define battery file ID range to process (1 to 8)
    BATTERY_IDS = range(1, 9) 
    # Filename template
    FILENAME_TEMPLATE = "/root/2025.6.19-阶段4尝试-研究一下何时停止/src/data/INR18650_{}_ui.npy"
    # Threshold for judging voltage. If a data point is greater than this value, we assume it's current (mA).
    VOLTAGE_THRESHOLD = 10.0 

    print("--- Starting to process UI data files ---")
    
    # Iterate through specified battery IDs
    for i in BATTERY_IDS:
        filename = FILENAME_TEMPLATE.format(i)
        
        # Check if file exists
        if not os.path.exists(filename):
            print(f"\nFile '{filename}' not found, skipping.")
            continue
            
        print(f"\n--- Processing: {filename} ---")
        
        try:
            # Load .npy file
            data = np.load(filename)
            
            # Basic shape validation
            if data.ndim != 3 or data.shape[1] != 2:
                print(f"  - Error: File '{filename}' data shape incorrect (should be (N, 2, 64)), skipped.")
                continue

            # ① Check and swap voltage and current order
            # We judge by checking the first cycle, first feature, first data point
            # If this value is greater than threshold (e.g., 10), we think it's milliampere-level current, need to swap
            if data[0, 0, 0] > VOLTAGE_THRESHOLD:
                print("  - Detected data order as [current, voltage], swapping to [voltage, current]...")
                # Use advanced indexing to swap two slices of axis 1
                data = data[:, [1, 0], :]
            else:
                print("  - Data order is [voltage, current], no swap needed.")

            # After previous step, we ensure data[:, 1, :] is now definitely current data
            if data[:, 1, :].mean() > 100:
                    # ② Convert current data from mA to A
                print("  - Converting current unit from mA to A (divide by 1000)...")
                data[:, 1, :] = data[:, 1, :] / 1000.0
                
                # ③ Overwrite save modified file
                np.save(filename, data)
                print(f"  - ✅ File '{filename}' successfully processed and overwritten.")
                
        except Exception as e:
            print(f"  - Error: Unexpected error occurred while processing file '{filename}': {e}")                                                       
            continue    





if __name__ == "__main__":#Main function is when __name__ = main, seems like only main process will execute
    
    # config_dir = "config/config_ecg.yaml"
    # config = yaml.load(open(config_dir, "r"), Loader=yaml.FullLoader)#Use predefined config file, load definitions into config

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Battery_list =  ['INR_4']
    # # read_capacity_data(Battery_list)
    # all_battery_capacity_list, all_battery_ui_list = load_all_data()

    change_ui()
    

    """
    if config['dataset']['path'] == '':#Indicates no npy file to load yet
        read_voltage_data(Battery_list,config)

    else:
        #Below is getting data from already saved py files, not needed for now
        all_battery_ui_list = load_ui_data(config)
    """









