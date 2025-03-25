import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy

def corio(lat):
    """Calculate the Coriolis parameter for a given latitude."""
    return 2 * (2 * np.pi / (24 * 60 * 60)) * np.sin(lat * (np.pi / 180))

def get_hist(y, k_mean, k_std):
    """Get histogram values for normalized data."""
    vals, binss = np.histogram(np.exp(y * k_std + k_mean), range=(0, 1.2), bins=100)
    return vals, 0.5 * (binss[0:-1] + binss[1:])

def get_hist2(y):
    """Get histogram values for error data."""
    vals, binss = np.histogram(y, range=(-0.2, 0.2), bins=100)
    return vals, 0.5 * (binss[0:-1] + binss[1:])

def performance_sigma_point(model, x, valid_x, y, valid_y, k_mean, k_std):
    """Plot the performance of a neural network model.

    Parameters:
        model: Trained neural network model.
        x: Training input data.
        valid_x: Validation input data.
        y: Training output data.
        valid_y: Validation output data.
        k_mean: Mean normalization values.
        k_std: Standard deviation normalization values.
    """
    # plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['font.size'] = 15
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'  # ensures it can math compatibility with symbols in your code without erroring fix no cursive_fontsystem


    y_pred_train = model(x)
    y_pred_test = model(valid_x)

    ycpu = y.cpu().detach().numpy()
    ytestcpu = valid_y.cpu().detach().numpy()
    yptraincpu = y_pred_train.cpu().detach().numpy()
    yptestcpu = y_pred_test.cpu().detach().numpy()

    ystd = np.zeros(16)
    yteststd = np.zeros(16)
    ypstd = np.zeros(16)
    ypteststd = np.zeros(16)
    yerr = np.zeros(16)
    kappa_mean = np.zeros(16)

    for i in range(16):
        ystd[i] = np.std(np.exp(ycpu[:, i] * k_std[i] + k_mean[i]))
        yteststd[i] = np.std(np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]))
        ypstd[i] = np.std(np.exp(yptraincpu[:, i] * k_std[i] + k_mean[i]))
        ypteststd[i] = np.std(np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i]))
        yerr[i] = np.std(np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]) - np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i]))

        kappa_mean[i] = np.mean(np.exp(ycpu[:, i] * k_std[i] + k_mean[i]))

    plt.figure(figsize=(15, 10))

    ind = np.arange(0, 16)
    ind_tick = np.arange(1, 17)[::-1]

    # Subplot 1: Boxplot of network output differences
    plt.subplot(1, 4, 1)
    for i in range(16):
        plt.boxplot(ytestcpu[:, i] - yptestcpu[:, i], vert=False, positions=[i], showfliers=False, whis=(5, 95), widths=0.5)
    plt.xlim([-2.0, 2.0])
    plt.yticks(ind, ind_tick)
    plt.title(r'(a) Output of network $\mathcal{N}_1$ ')
    plt.ylabel('Node')

    # Subplot 2: Boxplot of shape function differences
    plt.subplot(1, 4, 2)
    for i in range(16):
        plt.boxplot(kappa_mean[i] + np.exp(ytestcpu[:, i] * k_std[i] + k_mean[i]) - np.exp(yptestcpu[:, i] * k_std[i] + k_mean[i]),
                    vert=False, positions=[i], showfliers=False, whis=(5, 95), widths=0.5)
    plt.yticks([])
    plt.title(r'(b) Shape function $g(\sigma)$')
    plt.xlabel(r'$g(\sigma)$')

    # Subplots 3 & 4: Histograms
    k12 = 15
    for k in range(16):
        plt.subplot(16, 4, 4 * k + 3)
        vals, binss = get_hist(ytestcpu[:, k12], k_mean[k12], k_std[k12])
        plt.plot(binss, vals, color='blue')

        vals, binss = get_hist(yptestcpu[:, k12], k_mean[k12], k_std[k12])
        plt.plot(binss, vals, color='red')
        if k < 15:
            plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.title('(c) Probability density histogram')

        plt.subplot(16, 4, 4 * k + 4)
        vals, binss = get_hist2(np.exp(ytestcpu[:, k12] * k_std[k12] + k_mean[k12]) - np.exp(yptestcpu[:, k12] * k_std[k12] + k_mean[k12]))
        plt.plot(binss, vals, color='green')
        if k < 15:
            plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.title('(d) Error histogram ')

        k12 -= 1

    plt.tight_layout()
    print("Plot saved as 'modelstats.pdf'")

def performance_sigma_point_batched(model, x, valid_x, y, valid_y, k_mean, k_std, batch_size=1000):
    """Plot the performance of a Gaussian Process model with batched processing to reduce memory usage.
    
    Parameters:
        model: Trained Gaussian Process model.
        x: Training input data.
        valid_x: Validation input data.
        y: Training output data.
        valid_y: Validation output data.
        k_mean: Mean normalization values.
        k_std: Standard deviation normalization values.
        batch_size: Size of batches for processing, default is 1000.
    """
    
    # Set plotting parameters
    plt.rcParams['font.size'] = 15
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'  # Ensure compatibility with mathematical symbols
    
    # Initialize arrays for storing statistical data
    ystd = np.zeros(16)
    yteststd = np.zeros(16)
    ypstd = np.zeros(16)
    ypteststd = np.zeros(16)
    yerr = np.zeros(16)
    kappa_mean = np.zeros(16)
    
    # Prepare data storage for boxplot
    output_diff = [[] for _ in range(16)]
    shape_diff = [[] for _ in range(16)]
    
    # Prepare data storage for histograms
    # Only store data for nodes that need to be plotted
    hist_data_true = [[] for _ in range(16)]
    hist_data_pred = [[] for _ in range(16)]
    hist_data_error = [[] for _ in range(16)]
    
    # First calculate kappa_mean (average of training data)
    # Process training data in batches
    sum_exp_y = np.zeros(16)
    n_batches_train = (x.shape[0] + batch_size - 1) // batch_size
    
    # Disable gradient calculation to save memory
    with torch.no_grad():
        for b in range(n_batches_train):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, x.shape[0])
            
            # Get current batch
            batch_y = y[start_idx:end_idx]
            
            # Convert to CPU array
            batch_y_cpu = batch_y.cpu().numpy()
            
            # Accumulate exp(y) values
            for i in range(16):
                sum_exp_y[i] += np.sum(np.exp(batch_y_cpu[:, i] * k_std[i] + k_mean[i]))
                
            # Clear unnecessary variables to free memory
            del batch_y, batch_y_cpu
    
    # Calculate averages
    for i in range(16):
        kappa_mean[i] = sum_exp_y[i] / x.shape[0]
    
    # Process validation data in batches
    n_batches_valid = (valid_x.shape[0] + batch_size - 1) // batch_size
    
    # Collect prediction differences from all batches
    with torch.no_grad():
        for b in range(n_batches_valid):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, valid_x.shape[0])
            
            # Get current batch
            batch_x = valid_x[start_idx:end_idx]
            batch_y = valid_y[start_idx:end_idx]
            
            # Model prediction - Gaussian process model returns MultitaskMultivariateNormal
            batch_y_pred = model(batch_x)
            
            # Get prediction mean - this is the method applicable to Gaussian process models
            batch_y_pred_mean = batch_y_pred.mean
            
            # Convert to CPU arrays
            batch_y_cpu = batch_y.cpu().numpy()
            batch_y_pred_cpu = batch_y_pred_mean.cpu().detach().numpy()
            
            # Update statistics
            for i in range(16):
                # Calculate true and predicted values
                true_values = np.exp(batch_y_cpu[:, i] * k_std[i] + k_mean[i])
                pred_values = np.exp(batch_y_pred_cpu[:, i] * k_std[i] + k_mean[i])
                
                # Calculate differences
                diff = batch_y_cpu[:, i] - batch_y_pred_cpu[:, i]
                shape_diff_values = true_values - pred_values
                
                # Store differences for boxplot
                output_diff[i].extend(diff.tolist())
                shape_diff[i].extend(shape_diff_values.tolist())
                
                # Sample data for histograms (k12 represents reversed node index)
                k12 = 15 - i
                if 0 <= k12 < 16:  # Ensure index is valid
                    # Randomly select a subset of samples from current batch to avoid storing all data
                    sample_size = min(50, len(batch_y_cpu))
                    sample_indices = np.random.choice(len(batch_y_cpu), sample_size, replace=False)
                    
                    hist_data_true[k12].extend(batch_y_cpu[sample_indices, k12].tolist())
                    hist_data_pred[k12].extend(batch_y_pred_cpu[sample_indices, k12].tolist())
                    hist_data_error[k12].extend(shape_diff_values[sample_indices].tolist())
            
            # Clear unnecessary variables to free memory
            del batch_x, batch_y, batch_y_pred, batch_y_pred_mean, batch_y_cpu, batch_y_pred_cpu
    
    # Calculate final statistics
    for i in range(16):
        # Convert to numpy arrays for calculation
        output_diff_array = np.array(output_diff[i])
        shape_diff_array = np.array(shape_diff[i])
        
        # Calculate standard deviations
        ystd[i] = np.std(output_diff_array)
        yerr[i] = np.std(shape_diff_array)
        
        # Clear large arrays to free memory
        del output_diff_array, shape_diff_array
    
    # Create figure
    plt.figure(figsize=(15, 10))
    ind = np.arange(0, 16)
    ind_tick = np.arange(1, 17)[::-1]
    
    # Subplot 1: Boxplot of network output differences
    plt.subplot(1, 4, 1)
    for i in range(16):
        plt.boxplot(output_diff[i], vert=False, positions=[i], showfliers=False, whis=(5, 95), widths=0.5)
    plt.xlim([-2.0, 2.0])
    plt.yticks(ind, ind_tick)
    plt.title(r'(a) Output of network $\mathcal{N}_1$ ')
    plt.ylabel('Node')
    
    # Subplot 2: Boxplot of shape function differences
    plt.subplot(1, 4, 2)
    for i in range(16):
        plt.boxplot(kappa_mean[i] + shape_diff[i], vert=False, positions=[i], showfliers=False, whis=(5, 95), widths=0.5)
    plt.yticks([])
    plt.title(r'(b) Shape function $g(\sigma)$')
    plt.xlabel(r'$g(\sigma)$')
    
    # Subplots 3 and 4: Histograms
    for k in range(16):
        k12 = 15 - k  # Reverse index
        
        # Subplot 3: Probability density histogram
        plt.subplot(16, 4, 4 * k + 3)
        if hist_data_true[k12]:  # Ensure data exists
            vals, binss = get_hist(np.array(hist_data_true[k12]), k_mean[k12], k_std[k12])
            plt.plot(binss, vals, color='blue')
            vals, binss = get_hist(np.array(hist_data_pred[k12]), k_mean[k12], k_std[k12])
            plt.plot(binss, vals, color='red')
        
        if k < 15:
            plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.title('(c) Probability density histogram')
        
        # Subplot 4: Error histogram
        plt.subplot(16, 4, 4 * k + 4)
        if hist_data_error[k12]:  # Ensure data exists
            vals, binss = get_hist2(np.array(hist_data_error[k12]))
            plt.plot(binss, vals, color='green')
        
        if k < 15:
            plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.title('(d) Error histogram ')
    
    plt.tight_layout()
    plt.savefig('modelstats.pdf')
    print("Plot saved as 'modelstats.pdf'")