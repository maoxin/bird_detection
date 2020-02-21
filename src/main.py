import subprocess

# normal model ex
## ex1
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

## ex2
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

## ex3
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

## ex4
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

# attention model ex
## ex1
def exa1():
    # baseline, attention, no aug
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention',
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
        "/media/data1/mx_model/bird_detection/bird_detection/exa1",
    ]

    subprocess.run(command)

def exa2():
    # baseline, attention, aug
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention',
        "--use-aug",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exa2",
    ]

    subprocess.run(command)

def exa3():
    # baseline, attention, aug, 32 parts
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention',
        "--num-parts",
        "32",
        "--use-aug",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exa3",
    ]

    subprocess.run(command)

def exa4():
    # baseline, attention, aug, 32 parts
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention',
        "--num-parts",
        "4",
        "--use-aug",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exa4",
    ]

    subprocess.run(command)

# def exa2():
#     # ft baseline
#     command = [
#         "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
#         "-m",
#         "torch.distributed.launch",
#         "--nproc_per_node=2",
#         "--use_env",
#         "train_bird_detection.py",
#         "--model",
#         'attention',
#         "--dataset",
#         "synthesized",
#         "--small-set",
#         "--epochs",
#         "26",
#         "--lr",
#         "0.005",
#         "--lr-steps",
#         "16",
#         "22",
#         "--aspect-ratio-group-factor",
#         "3",
#         "--print-freq",
#         "5",
#         "--output-dir",
#         "/media/data1/mx_model/bird_detection/bird_detection/exa2",
#     ]

#     subprocess.run(command)

# def exa4():
#     # ft baseline
#     command = [
#         "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
#         "-m",
#         "torch.distributed.launch",
#         "--nproc_per_node=2",
#         "--use_env",
#         "train_bird_detection.py",
#         "--model",
#         'attention',
#         "--dataset",
#         "real",
#         "--epochs",
#         "26",
#         "--lr",
#         "0.005",
#         "--lr-steps",
#         "16",
#         "22",
#         "--aspect-ratio-group-factor",
#         "3",
#         "--print-freq",
#         "5",
#         "--output-dir",
#         "/media/data1/mx_model/bird_detection/bird_detection/exa4",
#         "--resume",
#         "/media/data1/mx_model/bird_detection/bird_detection/exa2/model_25.pth",
#         "--ft",
#     ]

#     subprocess.run(command)

## ex5
def exa5():
    # ft baseline using original model state
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention',
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
        "/media/data1/mx_model/bird_detection/bird_detection/exa4",
        "--resume",
        "/media/data1/mx_model/bird_detection/bird_detection/ex1/model_25.pth",
        "--ft",
    ]

    subprocess.run(command)

def exa6():
    # baseline, attention, no aug, FocalLoss
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention',
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
        "/media/data1/mx_model/bird_detection/bird_detection/exa6",
    ]

    subprocess.run(command)

def exa7():
    # baseline, attention, no aug, FocalLoss gamma5
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention',
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
        "/media/data1/mx_model/bird_detection/bird_detection/exa7",
    ]

    subprocess.run(command)

def exa8():
    # baseline, attention, no aug, FocalLoss gamma2
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention',
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
        "/media/data1/mx_model/bird_detection/bird_detection/exa8",
    ]

    subprocess.run(command)

# egret version
def exae1():
    # baseline, attention, no aug, FocalLoss gamma2
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention',
        "--num-classes",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exae1",
    ]

    subprocess.run(command)

def exae2():
    # baseline, attention, no aug, FocalLoss gamma0
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention',
        "--num-classes",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exae2",
    ]

    subprocess.run(command)

# attention transformer model ex
## ex1
def exat1():
    # baseline, no aug, 8 parts, layer 3
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention_transformer',
        "--num-parts",
        "8",
        # "--use-aug",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exat1",
    ]

    subprocess.run(command)

def exat2():
    # baseline, no aug, 8 parts, layer 1
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention_transformer',
        "--num-parts",
        "8",
        # "--use-aug",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exat2",
    ]

    subprocess.run(command)

def exat2():
    # baseline, no aug, 8 parts, layer 1
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention_transformer',
        "--num-parts",
        "8",
        # "--use-aug",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exat2",
    ]

    subprocess.run(command)

def exat3():
    # baseline, aug, 8 parts, layer 1
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--model",
        'attention_transformer',
        "--num-parts",
        "8",
        "--use-aug",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exat3",
    ]

    subprocess.run(command)

# def exat1():
#     # ft baseline
#     command = [
#         "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
#         "-m",
#         "torch.distributed.launch",
#         "--nproc_per_node=2",
#         "--use_env",
#         "train_bird_detection.py",
#         "--model",
#         'attention_transformer',
#         "--dataset",
#         "real",
#         "--epochs",
#         "26",
#         "--lr",
#         "0.005",
#         "--lr-steps",
#         "16",
#         "22",
#         "--aspect-ratio-group-factor",
#         "3",
#         "--print-freq",
#         "5",
#         "--output-dir",
#         "/media/data1/mx_model/bird_detection/bird_detection/exat1",
#     ]

#     subprocess.run(command)

# normal model, only instance ex
def exi1():
    # real baseline
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--num-classes",
        "2",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exi1",
    ]

    subprocess.run(command)

def exi2():
    # syn baseline
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--num-classes",
        "2",
        "--dataset",
        "synthesized",
        "--small-set",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exi2",
    ]

    subprocess.run(command)

def exi3():
    # syn baseline, test on real
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--num-classes",
        "2",
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
        "--resume",
        "/media/data1/mx_model/bird_detection/bird_detection/exi2/model_25.pth",
        "--test-only",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/exi3",
    ]

    subprocess.run(command)

def exi4():
    # ft baseline
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--num-classes",
        "2",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exi4",
        "--resume",
        "/media/data1/mx_model/bird_detection/bird_detection/exi2/model_25.pth",
        "--ft",
    ]

    subprocess.run(command)

def exi5():
    # ft baseline, small lr
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--num-classes",
        "2",
        "--dataset",
        "real",
        "--only-instance",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exi5",
        "--resume",
        "/media/data1/mx_model/bird_detection/bird_detection/exi2/model_25.pth",
        "--ft",
    ]

    subprocess.run(command)

# coco baseline
def exc1():
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
        "/media/data1/mx_model/bird_detection/bird_detection/exc1",
        "--test-only",
    ]

    subprocess.run(command)

def exc2():
    # real baseline, train
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exc2",
    ]

    subprocess.run(command)

def exc3():
    # real baseline, train, small lr
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_detection.py",
        "--dataset",
        "real",
        "--only-instance",
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
        "/media/data1/mx_model/bird_detection/bird_detection/exc3",
    ]

    subprocess.run(command)


if __name__ == "__main__":
    exae2()