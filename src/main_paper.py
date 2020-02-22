import subprocess

def ex1_p():
    # baseline
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--use-focal-loss",
        "--focal-gamma",
        "0",
        "--dataset",
        "real",
        "--epochs",
        "26",
        "--lr",
        "0.005",
        "--lr-steps",
        "16",
        "22",
        "--aspect-ratio-group-factor",
        "3",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/ex1_p",
    ]

    subprocess.run(command)

def ex2_p():
    # baseline, atten
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        "attention",
        "--use-focal-loss",
        "--focal-gamma",
        "0",
        "--dataset",
        "real",
        "--epochs",
        "26",
        "--lr",
        "0.005",
        "--lr-steps",
        "16",
        "22",
        "--aspect-ratio-group-factor",
        "3",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/ex2_p",
    ]

    subprocess.run(command)

def ex3_p():
    # baseline, focal-loss == cross entropy
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--use-focal-loss",
        "--focal-gamma",
        "1",
        "--dataset",
        "real",
        "--epochs",
        "26",
        "--lr",
        "0.005",
        "--lr-steps",
        "16",
        "22",
        "--aspect-ratio-group-factor",
        "3",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/ex3_p",
    ]

    subprocess.run(command)

def ex4_p():
    # baseline, focal-loss == cross entropy
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--use-focal-loss",
        "--focal-gamma",
        "2",
        "--dataset",
        "real",
        "--epochs",
        "26",
        "--lr",
        "0.005",
        "--lr-steps",
        "16",
        "22",
        "--aspect-ratio-group-factor",
        "3",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/ex4_p",
    ]

    subprocess.run(command)

def ex5_p():
    # baseline, focal-loss == cross entropy
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--use-focal-loss",
        "--focal-gamma",
        "3",
        "--dataset",
        "real",
        "--epochs",
        "26",
        "--lr",
        "0.005",
        "--lr-steps",
        "16",
        "22",
        "--aspect-ratio-group-factor",
        "3",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/ex5_p",
    ]

    subprocess.run(command)

if __name__ == "__main__":
    ex2_p()