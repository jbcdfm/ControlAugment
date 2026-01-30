from datetime import datetime


    
def log_file_initiate(filename: str, title: str = "Log File", loginfo: dict = None):
    """
    Creates a log file and writes an initial header with a timestamp, and adds 
    information incl. dataset, model type, and augmentation method.
    
    :param filename: Name of the log file (e.g., "log.txt").
    :param title: Title of the log file.
    :param loginfo: Dictionary of metadata to include in the log header.
    """
    with open(filename, "w") as log_file:
        log_file.write(f"# {title}\n")
        log_file.write(f"# Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("\n")  # Add a blank line for readability
        
        if loginfo:
            log_file.write("# Info:\n")
            for key, value in loginfo.items():
                log_file.write(f"#   {key}: {value}\n")
        
        log_file.write("\n")  # Add a blank line for readability
    
def log_run_result(folder_and_file: str, run, acc_val, acc_test, acc_test_TTA, kappa, lr, Gamma, alpha, N_p):
    epochs = [1 + x for x in range(len(acc_val))]
    # Append data to log file
    with open(folder_and_file, "a") as log_file:
        log_file.write("----------------------------------------------------------\n")
        log_file.write(f"Run {run+1} data: \n")
        log_file.write("----------------------------------------------------------\n")
        for e in epochs:
            if e%N_p==0:
                log_file.write(f"Epoch {e}: ")
                log_file.write(f"Classification val. accuracy: {acc_val[e-1]:.3f} % | kappa = {kappa[int((e-N_p)/N_p)]:.3f} | Learning rate: {lr[e-1]:.6f}\n")
                log_file.write("Gamma = ")
                for gam in Gamma[e-1]:
                    log_file.write(f"{gam:.3f}, ")
                log_file.write("\n")
                log_file.write("alpha_IA = ")
                for alp in alpha[e-1]:
                    log_file.write(f"{alp:.3f}, ")
                log_file.write("\n")
                # log_file.write(f"Learning rate: {lr[e-1]:.6f}\n")
                log_file.write('-----------------------------\n')
        log_file.write('-----------------------------\n')
        log_file.write(f"Final accuracy of run {run+1}: {acc_test:.3f} %  (with TTA: {acc_test_TTA:.3f} %) \n")
        log_file.write('-----------------------------\n')
        log_file.write('----------------------------------------------------------\n')
        
        
def log_summary(folder_and_file: str, Final_correct, Final_correct_TTA, Final_correct_val):
    with open(folder_and_file, "a") as log_file:
        log_file.write('----------------------------------------------------------\n')
        log_file.write("Summary of classification test accuracies of all runs:\n")
        for correct in Final_correct:
            log_file.write(f"{correct:.3f}\n")
        
        log_file.write('----------------------------------------------------------\n')
        log_file.write("Summary of classification test accuracies of all runs (with TTA):\n")
        for correct in Final_correct_TTA:
            log_file.write(f"{correct:.3f}\n")
            
        log_file.write('----------------------------------------------------------\n')
        log_file.write("Summary of classification validation accuracies:\n")
        for correct in Final_correct_val:
            log_file.write(f"{correct:.3f}\n")