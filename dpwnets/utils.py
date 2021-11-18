from datetime import datetime

def log_msg(message):
    print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)

def error_msg(message):
    print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'ERROR >>>>>>>>>>>>>>> ' + message + ' <<<<<<<<<<<<<<<')

def check_progress(i, total):
    progress_check = [int(1/5*total), int(2/5*total), int(3/5*total), int(4/5*total)]
    if i in progress_check:
        log_msg("{:.0%}".format(i/total)) 