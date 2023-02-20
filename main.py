import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import torch
from torch.utils.data import random_split, DataLoader

from model import *

MANUAL_SEED = 1145141919810
BATCH_SIZE = 5
EPOCH = 200

SPECIAL_CHAR = {'ï': 'i'}


def word_one_hot(word: str):
    """
    :param word:
    :return: numpy array in shape(5, 26)
    """
    word = word.replace(" ", "").lower()
    if len(word) != 5:
        raise ValueError("The length of word is not 5.")
    one_hot = torch.zeros((5, 26), dtype=torch.long)
    for i, char in enumerate(word):
        c = ord(char) - ord('a')
        if not(0 <= c <= 25): # 如果不是ASCII的话
            if char in SPECIAL_CHAR:
                c = ord(SPECIAL_CHAR[char]) - ord('a')
            else:
                raise UnicodeError("{} is a special char. \
                You can mannually add it into SPECIAL_CHAR to avoid Exception.".format(char))
        one_hot[i, c] = 1
    return one_hot


def tris_to_label(*args):
    assert len(args) == 7
    label = np.array(args)
    label = label / np.sum(label)
    label = torch.from_numpy(label)
    label = label.float()
    return label


def load_file():
    """
    one_hot and label
    :return: [torch[long], torch[float32]]
    """
    # 必须用utf的方式另存为, 因为有特殊的一些字符
    df = pd.read_csv("./Word and Tries.csv")
    dataset = []
    for row in df.itertuples():
        tmp = list(row)
        if tmp[0] == 20:
            print(tmp[1])
        try:
            dataset.append([word_one_hot(tmp[1]),
                            tris_to_label(*tmp[-7:])])
        except ValueError:
            continue  # 长度不对跳过
        except AssertionError:
            continue
    return dataset


if __name__ == '__main__':
    dataset = load_file()

    train_data, test_data = random_split(dataset, [0.8, 0.2],
                                         generator=torch.Generator().manual_seed(MANUAL_SEED))
    print("Train Length: {}".format(len(train_data)))
    print("Test Length: {}".format(len(test_data)))

    # 时序分析是不能打乱
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    # 引入模型
    model = Model()

    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss() # 不太行

    # 1e-(4~6)
    learning_rate = 1e-5
    """
    MSELoss下：
    -4 r2 0.9 50轮内, 但波动
    -6 r2 0.9 400轮以上, 但loss也不够低
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 监视
    train_loss = []
    test_loss = []
    r2 = []

    # total_train_step = 0
    # total_train = []
    # total_test_step = 0
    # total_test = []

    r2_cnt = 0

    for i in range(EPOCH):
        print("Epoch {}".format(i + 1))

        model.train()
        total_train_loss = 0
        cnt = 0
        for one_hot, tries in train_dataloader:
            output = model(one_hot)
            loss = loss_fn(output, tries)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            cnt += 1
            # total_train_step += 1
            # total_train.append(loss.item())
        train_loss.append(total_train_loss / cnt)

        # 评估
        model.eval()
        total_test_loss = 0
        total_r2 = 0
        cnt = 0
        with torch.no_grad():
            for one_hot, tries in train_dataloader:
                output = model(one_hot)
                loss = loss_fn(output, tries)
                total_test_loss += loss.item()

                # total_test_step += 1
                # total_test.append(loss.item())

                tries_np = np.array(tries)
                output_np = np.array(output)
                tmp_r2 = 0
                batch = output_np.shape[0]
                for j in range(batch):
                    tmp_r2 += r2_score(tries_np[j], output_np[j])
                    r2_cnt += 1
                    if r2_cnt % 5000 == 0:
                        plt.scatter(tries_np[j], output_np[j])
                        plt.show()
                        x = np.arange(7)
                        plt.plot(x, tries_np[j], "g")
                        plt.plot(x, output_np[j], "y")
                        plt.show()
                        print("r2:{}".format(r2_score(tries_np[j], output_np[j])))
                total_r2 += tmp_r2 / batch
                cnt += 1
            test_loss.append(total_test_loss / cnt)
            r2.append(total_r2 / cnt)

        print("Train Loss:{} Test Loss:{} R2:{}".format(train_loss[-1],
                                                        test_loss[-1],
                                                        r2[-1]))
        if i % 10 == 0:
            length = len(train_loss)
            x = np.arange(length)
            plt.plot(x, np.array(train_loss), "b")
            plt.plot(x, np.array(test_loss), "r")
            plt.show()

            # plt.plot(x, np.array(r2), "r")
            # plt.show()

    print("-------------")
    oh = word_one_hot("kiosk")
    la = tris_to_label(0, 24, 170, 401, 313, 130, 19)
    model.eval()
    with torch.no_grad():
        output = model(oh.view(1, 5, 26))
        loss = loss_fn(output, la.view(1, -1))
        la = np.array(la).reshape(-1)
        ou = np.array(output).reshape(-1)
        print("Loss :{}".format(loss.item()))
        print("r2:{}".format(r2_score(la, ou)))
        x = np.arange(7)
        plt.plot(x, la, "g")
        plt.plot(x, ou, "y")
        plt.show()
