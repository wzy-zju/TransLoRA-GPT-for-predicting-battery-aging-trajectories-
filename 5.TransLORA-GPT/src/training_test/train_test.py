from src.utils import *
from torch.utils.tensorboard import SummaryWriter
import time


def train_test_one_step(config, model, optimizer, writer, data_pack,total_params, trainable_params, trainable_percentage):
    train_loader = data_pack['train_loader']
    val_loader = data_pack['val_loader']
    test_ui_windows = data_pack['test_x_ui_windows']
    test_soh_windows = data_pack['test_y_soh_windows']
    test_x_baseline = data_pack['test_x_baseline']
    
    train_losses , val_losses, track_cycles_seen = [], [], []
    cycles_seen , step = 0, 0

    patience = 20  # If validation loss doesn't improve for 20 consecutive epochs, stop training
    min_delta = 0.0001 # Loss must decrease by more than this value to be considered an improvement
    patience_counter = 0 # Counter
    best_val_loss = np.inf # Initialize best loss to positive infinity

    
    if config['train']:
        start_time = time.time()
        for epoch in range(3000):
            model.train()
            assert train_loader is not None, "train_loader is empty, please check data loading part!"
            for i, (x, y) in enumerate(train_loader):
                # torch.cuda.reset_peak_memory_stats(device=config['device'])

                x  = x.to(config["device"]).float()
                y = y.to(config["device"]).float()
                y_pred, _ = model(x, None)

                optimizer.zero_grad()
                loss = compute_policy_loss(y_pred, y, config)
                loss.backward()
                optimizer.step()
                cycles_seen += x.shape[0] * x.shape[1] # Number of batches, then number of windows per batch
                step  += 1

                # peak_memory_gb = torch.cuda.max_memory_reserved(device=config['device']) / (1024**3)
                # print(f"Epoch: {epoch}, Batch: {i}, Peak VRAM Usage: {peak_memory_gb:.4f} GB")

            # if step % config['eval_freq'] == 0:
            train_loss, val_loss, _ ,_ = evalute_model(model, train_loader, val_loader, config)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            writer.add_scalar('onestep_Loss/train', train_loss, epoch)
            writer.add_scalar('onestep_Loss/val', val_loss, epoch)
        


            if val_loss < best_val_loss - min_delta:
                # If there's improvement, update best loss, reset counter, and save current model as best model
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # If counter reaches patience value, trigger early stopping
            if patience_counter >= patience:
                print(f"Current epoch is {epoch}")
                break # Exit training loop
        end_time = time.time()
        elapsed_time = end_time - start_time

        save_model(model, config, val_loss)

        single_x_ui = test_ui_windows
        single_y_soh = test_soh_windows # Labels usually don't need to be put on GPU
        single_x_baseline = test_x_baseline
        loss, y_pred, y_label =  compute_test_loss(model, single_x_ui, single_y_soh, single_x_baseline, config)
        y_label = y_label.squeeze().to('cpu')
        y_pred = y_pred.squeeze().to('cpu')
        if config['smooth']:
            y_pred = smooth_pred(y_pred)

        # assert y_pred != None, "y_pred is empty"
        save_path, test_id, rmse = save_results_to_excel(y_pred, y_label, config, epoch, 
                                                         total_params, trainable_params, 
                                                         trainable_percentage, elapsed_time)
        plot_test(y_pred, y_label, config, save_path, test_id, rmse)

        writer.close()
    
    else:# Test mode
        model = load_model(model, config)
        num_test_samples = test_ui_windows.shape[0]

        for i in range(num_test_samples):
            single_x_ui = test_ui_windows[i:i+1]
            single_y_soh = test_soh_windows[i:i+1] # Labels usually don't need to be put on GPU
            single_x_baseline = test_x_baseline[i:i+1]
            loss, y_pred, y_label =  compute_test_loss(model, single_x_ui, single_y_soh, single_x_baseline, config)
            y_label = y_label.squeeze().to('cpu')
            y_pred = y_pred.squeeze().to('cpu')
            if config['smooth']:
                y_pred = smooth_pred(y_pred)
            # save_path, test_id, rmse = save_results_to_excel(y_pred, y_label, config,)
            # plot_test(y_pred, y_label, config, save_path, test_id, rmse)
            raise ValueError("test is not complete")



