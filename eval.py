import os.path
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from model import *
from main import word_one_hot, tris_to_label, WEIGHT_PATH

TEST_WORD = "kiosk"
TEST_TRIES = [0, 24, 170, 401, 313, 130, 19]

PREDICT_WORD = "EERIE"

STAT_NAME = "1_1.7400736396666616e-05.pth"


stat_pth = os.path.join(WEIGHT_PATH, STAT_NAME)
checkpoint = torch.load(stat_pth)

def run_model(one_hot):
    model = Model()
    model.eval()
    model.load_state_dict(checkpoint['net'])
    with torch.no_grad():
        return model(one_hot.view(1, 5, 26))

def eval_and_r2(true_tries, pred_tries): # torch (1 * -1)
    loss_fn = nn.MSELoss()
    loss = loss_fn(true_tries, pred_tries.view(1, -1))
    la = np.array(true_tries).reshape(-1)
    ou = np.array(pred_tries).reshape(-1)
    print("Loss :{}".format(loss.item()))
    print("r2:{}".format(r2_score(la, ou)))
    x = np.arange(7)
    plt.plot(x, la, "g")
    plt.plot(x, ou, "y")
    plt.show()

def predict_word(word: str):
    oh = word_one_hot(word)
    out = run_model(oh)
    y = np.array(out).reshape(-1)
    x = np.arange(7)
    plt.plot(x, y)
    plt.show()
    print("Tries of {} will be {}".format(word, y))


def test_more(word: str, tries_ls):
    oh = word_one_hot(word)
    la = tris_to_label(*tries_ls)
    output = run_model(oh)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, la.view(1, -1))
    la = np.array(la).reshape(-1)
    ou = np.array(output).reshape(-1)
    print("Loss :{}".format(loss.item()))
    print("r2:{}".format(r2_score(la, ou)))
    x = np.arange(7)
    plt.plot(x, la, "g")
    plt.plot(x, ou, "y")
    plt.show()


if __name__ == "__main__":
    predict_word(PREDICT_WORD)

    test_more(TEST_WORD, TEST_TRIES)
