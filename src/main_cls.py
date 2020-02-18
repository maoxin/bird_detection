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
        "train_bird_classification.py",
        "--dataset",
        "real",
        "--batch-size",
        "32",
        "--epochs",
        "26",
        "--lr",
        "0.005",
        "--lr-steps",
        "16",
        "22",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/cls_ex1",
    ]

    subprocess.run(command)

def ex2():
    # real baseline, attention transformer
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_classification.py",
        "--model",
        "attention_transformer",
        "--dataset",
        "real",
        "--batch-size",
        "32",
        "--epochs",
        "26",
        "--lr",
        "0.005",
        "--lr-steps",
        "16",
        "22",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/cls_ex2",
    ]

    subprocess.run(command)

def ex3():
    # real baseline, attention transformer
    command = [
        "/home/jz76/.local/share/virtualenvs/bird_detection-twxm2RaN/bin/python",
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=2",
        "--use_env",
        "train_bird_classification.py",
        "--model",
        "attention_transformer",
        "--dataset",
        "synthesized",
        "--small-set",
        "--batch-size",
        "50",
        "--epochs",
        "26",
        "--lr",
        "0.005",
        "--lr-steps",
        "16",
        "22",
        "--print-freq",
        "5",
        "--output-dir",
        "/media/data1/mx_model/bird_detection/bird_detection/cls_ex3",
    ]

    subprocess.run(command)



if __name__ == "__main__":
    ex3()