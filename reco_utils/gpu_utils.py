import subprocess

def get_gpu_name():
    try:
        from tensorflow.python.client import device_lib
        for device in device_lib.list_local_devices():
            if device.device_type == 'GPU':
                return device.physical_device_desc
        return "Aucun GPU détecté"
    except Exception as e:
        return f"Erreur : {e}"

def detect_gpu_type():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
        return output.decode().strip()
    except Exception as e:
        return "Unknown"
