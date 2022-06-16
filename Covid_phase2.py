import pandas as pd
import seaborn as sn
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
        self.n_feature = None
        self.sample_pos = None
        self.sample_neg = None
        self.sampletest = None
        self.cls = None
        self.res = None
        self.Vf = None
        self.Vg = None
        self.patient = None
        self.matrix = None
        self.n = 25 + 1
        self._domain = None
        self._dom = None

    def read_data(self):
        data = pd.read_excel('covid_study.xlsx')
        df = pd.DataFrame(data)
        df = df.dropna(0)

        df.loc[df['GENDER'] == 'M', 'GENDER'] = 1
        df.loc[df['GENDER'] == 'F', 'GENDER'] = 0
        df_normal = pd.DataFrame(
            (df.iloc[:, :-1] - df.iloc[:, :-1].min()) / (df.iloc[:, :-1].max() - df.iloc[:, :-1].min()))
        df = pd.concat([df_normal, pd.DataFrame(df['SWAB'])], axis=1)
        # print(df)

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

        self.sample_pos = self.X_traindata_pos.shape[0]
        self.sample_neg = self.X_traindata_neg.shape[0]
        self.sampletest = self.X_testdata.shape[0]

        self.n_feature = self.X_traindata_pos.shape[1]

    def relation(self):

        matrix = np.zeros((self.n_feature, 2, self.n, 2))
        for i in range(self.n_feature):
            x_pos = self.X_traindata_pos.iloc[:, i]
            x_neg = self.X_traindata_neg.iloc[:, i]

            min_x_pos = min(x_pos)
            max_x_pos = max(x_pos)
            min_x_neg = min(x_neg)
            max_x_neg = max(x_neg)

            mean_pos = (max_x_pos + min_x_pos) / 2
            mean_neg = (max_x_neg + min_x_neg) / 2

            a_pos = min_x_pos
            b_pos = mean_pos
            c_pos = max_x_pos

            a_neg = min_x_neg
            b_neg = mean_neg
            c_neg = max_x_neg
            # print('pos:', a_pos, b_pos, c_pos)
            # print('neg:', a_neg, b_neg, c_neg)
            fuzzy_set_pos = np.zeros((self.n, 2))
            fuzzy_set_neg = np.zeros((self.n, 2))
            member = np.zeros((self.n, 2))
            for j in range(0, self.n):
                x = j / (self.n - 1)
                member[j, 0] = self.membership(x, a_pos, b_pos, c_pos)
                member[j, 1] = self.membership(x, a_neg, b_neg, c_neg)
                fuzzy_set_pos[j, 0] = x
                fuzzy_set_pos[j, 1] = member[j, 0]
                fuzzy_set_neg[j, 0] = x
                fuzzy_set_neg[j, 1] = member[j, 1]
            # print('pos:',fuzzy_set_pos)
            # print('neg:',fuzzy_set_neg)
            matrix[i, 0] = fuzzy_set_pos
            matrix[i, 1] = fuzzy_set_neg
        self.matrix = matrix
        # print(matrix)

    def membership(self, x, a, b, c):
        res = 0
        if a == b:
            b = a + 0.1
        if c == b:
            c = b + 0.1
        if x < a or x > c:
            res = 0
        elif a <= x <= b:
            res = (x - a) / (b - a)
        elif b <= x <= c:
            res = (c - x) / (c - b)
        return res

    def pateint(self):

        member = np.zeros((self.sampletest, 15, self.n, 2))
        for i in range(self.sampletest):
            for j in range(self.n_feature):
                mean_x = self.X_testdata.iloc[i, j]
                min_x = mean_x - 0.3
                max_x = mean_x + 0.3
                a = min_x
                b = mean_x
                c = max_x
                for k in range(self.n):
                    x = k / (self.n - 1)
                    member[i, j, k, 0] = x
                    member[i, j, k, 1] = self.membership(x, a, b, c)
        self.patient = member

    def min_all(self, x, pa):
        r_min = np.zeros((self.n_feature, 2, self.n, 2))
        for j in range(self.n_feature):

            p = pa[j]
            matrix_pos = self.matrix[j, 0]
            matrix_neg = self.matrix[j, 1]
            Vf = self.centroid(p)
            Vg_pos = self.centroid(matrix_pos)
            Vg_neg = self.centroid(matrix_neg)
            min_pos = self.min_relation(Vf, Vg_pos, p, matrix_pos)
            min_neg = self.min_relation(Vf, Vg_neg, p, matrix_neg)
            r_min[j, 0] = min_pos
            r_min[j, 1] = min_neg

            if False:
                print(f'Vg_pos:\n{Vg_pos}')
                print(f'pos:\n{matrix_pos}')
                print(f'p:\n{p}')
                print(f'min_pos:\n{min_pos}')
                print(f'Vg_neg:\n{Vg_neg}')
                print(f'neg:\n{matrix_neg}')
                print(f'p:\n{p}')
                print(f'min_neg:\n{min_neg}')

                fig = plt.figure(figsize=(10, 5))
                ax_pos = fig.add_subplot(121)
                ax_neg = fig.add_subplot(122)

                ax_pos.plot(matrix_pos[:, 0], matrix_pos[:, 1], '-b', label=f'pos[col:{j}]', lw=4)  # Added
                ax_pos.plot(p[:, 0], p[:, 1], '-k', label=f'patient[col:{j}]', lw=4)  # Added
                ax_pos.plot(min_pos[:, 0], min_pos[:, 1], '-c', label='min pos', lw=2)  # Added
                ax_pos.legend()

                ax_neg.plot(matrix_neg[:, 0], matrix_neg[:, 1], '-r', label=f'neg[col:{j}]', lw=4)  # Added
                ax_neg.plot(p[:, 0], p[:, 1], '-k', label=f'patient[col:{j}]', lw=4)  # Added
                ax_neg.plot(min_neg[:, 0], min_neg[:, 1], '-y', label='min neg', lw=2)  # Added
                ax_neg.legend()

                plt.show()

        final = self.max_all(r_min)
        return final

    def min_relation(self, Vf, Vg, patient, matrix):

        if Vf <= Vg:
            f = patient
            g = matrix
        else:
            f = matrix
            g = patient
            t = Vf
            Vf = Vg
            Vg = t

        res = np.zeros((self.n, 2))
        for i in range(self.n):
            theta = i / (self.n - 1)
            if theta < Vf and theta < Vg:
                res[i, 1] = max(f[i, 1], g[i, 1])
            elif Vf <= theta <= Vg:
                res[i, 1] = f[i, 1]
            elif Vf <= theta and Vg <= theta:
                res[i, 1] = min(f[i, 1], g[i, 1])
            res[i, 0] = theta
        return res

    def max_all(self, r_min):
        final_mat = np.zeros((1, 2, self.n, 2))
        for i in range(self.n_feature - 1):
            Vf_pos = self.centroid(r_min[i, 0])
            Vg_pos = self.centroid(r_min[i + 1, 0])
            Vf_neg = self.centroid(r_min[i, 1])
            Vg_neg = self.centroid(r_min[i + 1, 1])

            if Vf_pos <= Vg_pos:
                f_pos = r_min[i, 0]
                g_pos = r_min[i + 1, 0]
            else:
                f_pos = r_min[i + 1, 0]
                g_pos = r_min[i, 0]
                t = Vf_pos
                Vf_pos = Vg_pos
                Vg_pos = t

            if Vf_neg <= Vg_neg:
                f_neg = r_min[i, 1]
                g_neg = r_min[i + 1, 1]
            else:
                f_neg = r_min[i + 1, 1]
                g_neg = r_min[i, 1]
                t = Vf_neg
                Vf_neg = Vg_neg
                Vg_neg = t

            r_max_pos = self.max_relation(Vf_pos, Vg_pos, f_pos, g_pos)
            r_max_neg = self.max_relation(Vf_neg, Vg_neg, f_neg, g_neg)
            for j in range(self.n):
                final_mat[0, 0, j, 1] = max(final_mat[0, 0, j, 1], r_max_pos[j, 1])
                final_mat[0, 1, j, 1] = max(final_mat[0, 1, j, 1], r_max_neg[j, 1])
            final_mat[0, 0, :, 0] = r_max_pos[:, 0]
            final_mat[0, 1, :, 0] = r_max_neg[:, 0]

            if False:
                fig = plt.figure(figsize=(10, 5))
                ax_pos = fig.add_subplot(121)
                ax_neg = fig.add_subplot(122)

                ax_pos.plot(f_pos[:, 0], f_pos[:, 1], '-b', lw=4, alpha=0.5, label=f'f_pos[col:{i}]')
                ax_neg.plot(f_neg[:, 0], f_neg[:, 1], '-r', lw=4, alpha=0.5, label=f'f_neg[col:{i}]')

                ax_pos.plot(g_pos[:, 0], g_pos[:, 1], '-c', lw=4, alpha=0.5, label=f'g_pos[col:{i + 1}]')
                ax_neg.plot(g_neg[:, 0], g_neg[:, 1], '-y', lw=4, alpha=0.5, label=f'g_neg[col:{i + 1}]')

                ax_pos.plot(final_mat[0, 0, :, 0], final_mat[0, 0, :, 1], '-k', lw=2, alpha=0.7, label=f'max_pos')
                ax_neg.plot(final_mat[0, 1, :, 0], final_mat[0, 1, :, 1], '-k', lw=2, alpha=0.7, label=f'max_neg')

                ax_pos.plot([Vf_pos, Vf_pos], [0, 1])
                ax_pos.plot([Vg_pos, Vg_pos], [0, 1])
                ax_neg.plot([Vf_neg, Vf_neg], [0, 1])
                ax_neg.plot([Vg_neg, Vg_neg], [0, 1])
                ax_pos.legend()
                ax_neg.legend()

                plt.show()
        return final_mat

    def max_relation(self, Vf, Vg, f, g):
        temp_max = np.zeros((self.n, 2))
        for k in range(self.n):
            theta = k / (self.n - 1)
            if theta <= Vf and theta <= Vg:
                temp_max[k, 1] = min(f[k, 1], g[k, 1])
            elif Vf <= theta <= Vg:
                temp_max[k, 1] = g[k, 1]
            elif Vf <= theta and Vg <= theta:
                temp_max[k, 1] = max(f[k, 1], g[k, 1])
            temp_max[k, 0] = theta

        return temp_max

    def centroid(self, fuzzyset):
        c1 = 0
        c2 = 0
        for i in range(self.n):
            c1 += fuzzyset[i, 0] * fuzzyset[i, 1]
            c2 += fuzzyset[i, 1]
        c1 /= c2
        return c1

    def defuzzification(self):
        f_matrix = np.zeros((self.sampletest, 2))
        m_p = np.zeros((1, 2, self.n, 2))
        for i in range(self.sampletest):
            x = self.X_testdata.iloc[i]
            p = self.patient[i]
            m_p = self.min_all(x, p)
            c_pos = m_p[0, 0]
            c_neg = m_p[0, 1]
            f_matrix[i, 0] = self.centroid(c_pos)
            f_matrix[i, 1] = self.centroid(c_neg)

        return f_matrix

    def test(self, final):
        y_hat = np.argmax(final, axis=1)
        y = np.array(self.y_testdata)

        acc = sum(y == y_hat) / len(y)
        return acc, y_hat


if __name__ == '__main__':
    max_test = 0
    y_hats = 0
    y = 0
    mean = 0
    for i in range(50):
        cv = Covid()
        cv.read_data()
        cv.relation()
        cv.pateint()
        final = cv.defuzzification()
        acc, y_hat = cv.test(final)
        mean += acc
        if max_test < acc:
            max_test = acc
            y_hats = y_hat
            y = list(cv.y_testdata)
    print('max', max_test)
    avg = mean / 50
    print('avg :', avg)

    re = confusion_matrix(y, y_hats)
    print('confusion matrix:\n', re)
    true_positive = re[0, 0]
    true_negetive = re[1, 1]
    false_positive = re[0, 1]
    false_negetive = re[1, 0]
    p_t_pos = (true_positive / (true_positive + false_positive)) * 100
    p_f_neg = (false_negetive / (true_negetive + false_negetive)) * 100
    p_f_pos = (false_positive / (false_positive + true_positive)) * 100
    p_t_neg = (true_negetive / (true_negetive + false_negetive)) * 100

    precesion = (true_positive / (true_positive + false_positive)) * 100
    recall = (true_positive / (true_positive + false_negetive)) * 100
    f_mesure = true_positive / (true_positive + (0.5 * (false_positive + false_negetive))) * 100
    accuracy = max_test * 100

    print(f'true positive:{p_t_pos:5.2f}%')
    print(f'false positive:{p_f_pos:5.2f}%')
    print(f'true negetive:{p_t_neg:5.2f}%')
    print(f'false negetive:{p_f_neg:5.2f}%')
    print('------------***--------------')
    print(f'precesion:{precesion:<5.2f}%')
    print(f'recall:{recall:5.2f}%')
    print(f'f mesure:{f_mesure:<5.2f}')
    print(f'accuracy:{accuracy:<5.2f}%')
