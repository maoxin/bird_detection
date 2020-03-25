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

def ex1s_p():
    # baseline, syn
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
        "synthesized",
        "--small-set",
        "--epochs",
        "13",
        "--lr",
        "0.005",
        "--lr-steps",
        "8",
        "11",
        "--aspect-ratio-group-factor",
        "3",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/ex1s_p",
    ]

    subprocess.run(command)

def ex1ft_p():
    # baseline, syn
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
        "0.0005",
        "--lr-steps",
        "16",
        "22",
        "--aspect-ratio-group-factor",
        "3",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/ex1ft_p",
        "--ft",
        "--resume",
        "/media/data1/mx_model/bird_detection/bird_detection/ex1s_p/model_12.pth",
    ]

    subprocess.run(command)

def exa1s_p():
    # baseline, syn
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
        "synthesized",
        "--small-set",
        "--epochs",
        "13",
        "--lr",
        "0.005",
        "--lr-steps",
        "8",
        "11",
        "--aspect-ratio-group-factor",
        "3",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/exa1s_p",
    ]

    subprocess.run(command)

def exa1ft_p():
    # baseline, syn
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
        "/media/data1/mx_model/bird_detection/bird_detection/exa1ft_p",
        "--ft",
        "--resume",
        "/media/data1/mx_model/bird_detection/bird_detection/exa1s_p/model_12.pth",
    ]

    subprocess.run(command)

def exa1fc_p():
    # baseline, attention, fc6, fc7
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
        "/media/data1/mx_model/bird_detection/bird_detection/exa1fc_p",
    ]

    subprocess.run(command)

def ex1itml_p():
    # baseline, attention, fc6, fc7
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
        "/media/data1/mx_model/bird_detection/bird_detection/ex1itml_p",
    ]

    subprocess.run(command)

def exf1s_p():
    # baseline, syn
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
        "synthesized",
        "--small-set",
        "--epochs",
        "13",
        "--lr",
        "0.005",
        "--lr-steps",
        "8",
        "11",
        "--aspect-ratio-group-factor",
        "3",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/exaf1s_p",
    ]

    subprocess.run(command)

def exf1ft_p():
    # baseline, syn
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
        "/media/data1/mx_model/bird_detection/bird_detection/exaf1ft_p",
        "--ft",
        "--resume",
        "/media/data1/mx_model/bird_detection/bird_detection/exaf1s_p/model_12.pth",
    ]

    subprocess.run(command)



def ex1i_p():
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
        "--only-instance",
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
        "/media/data1/mx_model/bird_detection/bird_detection/ex1i_p",
    ]

    subprocess.run(command)

def ex1si_p():
    # baseline, syn
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
        "synthesized",
        "--small-set",
        "--only-instance",
        "--epochs",
        "13",
        "--lr",
        "0.005",
        "--lr-steps",
        "8",
        "11",
        "--aspect-ratio-group-factor",
        "3",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/ex1si_p",
    ]

    subprocess.run(command)

def ex1fti_p():
    # baseline, syn
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
        "--only-instance",
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
        "/media/data1/mx_model/bird_detection/bird_detection/ex1fti_p",
        "--ft",
        "--resume",
        "/media/data1/mx_model/bird_detection/bird_detection/ex1si_p/model_12.pth",
    ]

    subprocess.run(command)

if __name__ == "__main__":
    for i in range(10):
        # exa1s_p()
        # exa1ft_p()

        # exf1s_p()
        # exf1ft_p()

        # ex1_p()
        # ex1s_p()
        # ex1ft_p()

        # ex2_p()
        # exa1ft_p()

        # ex1i_p()
        # ex1si_p()
        # ex1fti_p()

        # ex1_p()
        # ex1s_p()
        ex1ft_p()

        break