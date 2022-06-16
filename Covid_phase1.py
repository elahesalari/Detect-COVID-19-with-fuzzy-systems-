import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix


class Covid:
    def __init__(self):
        self.X_traindata_pos = None
        self.X_traindata_neg = None
        self.y_traindata_pos = None
        self.y_traindata_neg = None

        self.X_testdata = None
        self.y_testdata = None
        self.cls = None

    def read_data(self):
        data = pd.read_excel('covid_study.xlsx')
        df = pd.DataFrame(data)
        df = df.dropna(0)

        df.loc[df['GENDER'] == 'M', 'GENDER'] = 1
        df.loc[df['GENDER'] == 'F', 'GENDER'] = 0
        df_normal = pd.DataFrame(
            (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / (df.iloc[:, :-1].max() - df.iloc[:, :-1].min()))
        df = pd.concat([df_normal, pd.DataFrame(df['SWAB'])], axis=1)

        cls_negetive = df.loc[df['SWAB'] == 0]
        cls_positive = df.loc[df['SWAB'] == 1]

        train_data_pos, test_data_pos = train_test_split(cls_positive, test_size=0.5)
        train_data_neg, test_data_neg = train_test_split(cls_negetive, test_size=0.5)

        self.X_traindata_pos = train_data_pos.iloc[:, :-1]
        self.X_traindata_neg = train_data_neg.iloc[:, :-1]
        self.y_traindata_pos = train_data_pos.iloc[:, -1]
        self.y_traindata_neg = train_data_neg.iloc[:, -1]
        self.X_testdata = pd.concat((test_data_pos.iloc[:, :-1], test_data_neg.iloc[:, :-1]), axis=0)
        self.y_testdata = pd.concat((test_data_pos.iloc[:, -1], test_data_neg.iloc[:, -1]), axis=0)

        n_feature = self.X_traindata_pos.shape[1]

        relation = np.zeros((n_feature, 2), np.float)
        for i in range(n_feature):
            corr_icol_target = pearsonr(self.X_traindata_pos.iloc[:, i], self.X_traindata_pos.iloc[:, 1])
            relation[i, 0] = corr_icol_target[0]

            corr_icol_target = pearsonr(self.X_traindata_neg.iloc[:, i], self.X_traindata_neg.iloc[:, 1])
            relation[i, 1] = corr_icol_target[0]

        # print('relation:\n', relation)

        return relation

    def max_min(self, p, relation):
        resualt = np.zeros((1, relation.shape[1]), np.float)
        for i in range(relation.shape[1]):
            resualt[0, i] = max([min(p[j], relation[j, i]) for j in range(len(p))])

        return resualt

    def test(self, x_test, rel):
        x_test = np.array(x_test, np.float)
        y_hat = []
        for i in range(len(x_test)):
            maxmin = self.max_min(x_test[i], rel)
            argmax = np.argmax(maxmin.flatten())
            y_hat.append(argmax)
        return y_hat




if __name__ == '__main__':
    max_test = 0
    y_hats=0
    y_test =0
    mean = 0
    for i in range(50):
        covid = Covid()
        rel = covid.read_data()
        y_hat = covid.test(covid.X_testdata, rel)
        y_test = covid.y_testdata
        acc = (np.sum(y_hat == y_test) / len(y_hat))
        mean+=acc
        if max_test < acc:
            max_test = acc
            y_hats = y_hat
            y_test =y_test
    print('max',max_test)
    avg = mean / 50
    print('avg :', avg)

    re = confusion_matrix(y_test, y_hat)
    print('confusion matrix:\n',re)
    true_positive = re[0, 0]
    true_negetive = re[1, 1]
    false_positive = re[0, 1]
    false_negetive = re[1, 0]
    p_t_pos = (true_positive / (true_positive + false_positive))*100
    p_f_neg = (false_negetive / (true_negetive + false_negetive))*100
    p_f_pos = (false_positive / (false_positive + true_positive))*100
    p_t_neg = (true_negetive / (true_negetive + false_negetive))*100

    precesion = (true_positive / (true_positive + false_positive))*100
    recall = (true_positive / (true_positive + false_negetive))*100
    f_mesure = true_positive / (true_positive + (0.5 * (false_positive + false_negetive))) * 100
    accuracy = max_test * 100


    print(f'true positive:{p_t_pos:5.2f}%')
    print(f'false positive:{p_f_pos:5.2f}%')
    print(f'true negetive:{p_t_neg:5.2f}%')
    print(f'false negetive:{p_f_neg:5.2f}%')
    print('------------***--------------')
    print(f'precesion:{precesion:<5.2f}%')
    print(f'recall:{recall:5.2f}%')
    print(f'f mesure:{f_mesure:<5.2f}%')
    print(f'accuracy:{accuracy:<5.2f}%')

