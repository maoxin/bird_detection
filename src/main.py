import subprocess

# ex1
def ex1():
    # real baseline
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
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
        "/media/data1/mx_model/bird_detection/bird_detection/ex0",
    ]

    subprocess.run(command)

def ex2():
    # synthesized baseline
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--dataset",
        "synthesized",
        "--small-set",
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
        "/media/data1/mx_model/bird_detection/bird_detection/ex1",
    ]

    subprocess.run(command)

def ex3():
    # synthesized baseline, test on real dataset
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
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
        "/media/data1/mx_model/bird_detection/bird_detection/ex1",
        "--resume",
        "/media/data1/mx_model/bird_detection/bird_detection/ex1/model_25.pth",
        "--test-only",
    ]

    subprocess.run(command)

def ex4():
    # ft baseline
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
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
        "/media/data1/mx_model/bird_detection/bird_detection/ex3",
        "--resume",
        "/media/data1/mx_model/bird_detection/bird_detection/ex1/model_25.pth",
        "--ft",
    ]

    subprocess.run(command)

if __name__ == "__main__":
    ex4()