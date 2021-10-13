from datetime import datetime

def log(message):
    print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)

def error(message):
    print (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'ERROR >>>>>>>>>>>>>>> ' + message + ' <<<<<<<<<<<<<<<')