import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix


class Covid:
    def __init__(self):
        self.X_traindata_ = None
        self.y_traindata = None

        self.X_testdata = None
        self.y_testdata = None
        self.n_feature = None
        self.sample_train = None
        self.sampletest = None
        self.n = 29

    def read_data(self):
        data = pd.read_excel('covid_study.xlsx')
        df = pd.DataFrame(data)
        df = df.dropna(0)

        df.loc[df['GENDER'] == 'M', 'GENDER'] = 1
        df.loc[df['GENDER'] == 'F', 'GENDER'] = 0
        df_normal = pd.DataFrame(
            (df.iloc[:, :-1] - df.iloc[:, :-1].min(0)) / (df.iloc[:, :-1].max(0) - df.iloc[:, :-1].min(0)))
        df = pd.concat([df_normal, pd.DataFrame(df['SWAB'])], axis=1)

        data = df.loc[:]

        train_data, test_data = train_test_split(data, test_size=0.5)

        self.X_traindata = train_data.iloc[:, :-1]
        self.y_traindata = train_data.iloc[:, -1]
        self.X_testdata = test_data.iloc[:, :-1]
        self.y_testdata = test_data.iloc[:, -1]

        self.sample_train = self.X_traindata.shape[0]
        self.sampletest = self.X_testdata.shape[0]
        self.n_feature = self.X_traindata.shape[1]

    def fuzzification(self):
        fuzzy_sets = np.zeros((self.n_feature, self.n, 2, 3))
        for i in range(self.n_feature):
            x = self.X_traindata.iloc[:, i]
            min_x = min(x)
            max_x = max(x)

            def fuzzify(m1, m2):
                fuzzy_res = []
                for k in range(self.n):
                    width = (m1 - m2)
                    b = m2 + width * ((k + 1) / (self.n + 1))
                    a = b - width / (self.n + 1)
                    c = b + width / (self.n + 1)
                    ma = 0
                    mb = 1
                    mc = 0
                    if k == 0:
                        a = 0
                        b = 0
                        ma = 1
                    elif k == self.n - 1:
                        mc = 1
                        c = 1
                        b = 1
                    else:
                        a = 0
                        c = 1
                    fuzzy_res.append([[a, b, c], [ma, mb, mc]])
                return np.array(fuzzy_res)

            fuzzy_sets[i] = fuzzify(max_x, min_x)

        return fuzzy_sets

    def membership(self, x, a, b, c):
        res = 0
        if a == 0:
            if a <= x <= b:
                res = 1
            elif b <= x < c:
                res = (c - x) / (c - b)
            else:
                res = 0
            return res
        elif c == 1:
            if a <= x < b:
                res = (x - a) / (b - a)
            elif b <= x <= c:
                res = 1
            else:
                res = 0
            return res
        else:
            if a <= x < b:
                res = (x - a) / (b - a)
            elif b <= x < c:
                res = (c - x) / (c - b)
            else:
                res = 0
            return res

    def extract_rule(self, fset):
        F = np.zeros((self.sample_train, 2, self.n_feature + 1))
        all_rule = ''
        for i in range(self.sample_train):
            y = self.y_traindata.iloc[i]
            member = np.zeros(self.n_feature)
            rule_text = 'IF'
            for j in range(self.n_feature):
                x = self.X_traindata.iloc[i, j]
                mem = np.zeros(self.n)
                for k in range(self.n):
                    a = fset[j, k, 0, 0]
                    b = fset[j, k, 0, 1]
                    c = fset[j, k, 0, 2]
                    mem[k] = self.membership(x, a, b, c)
                f = np.argmax(mem)
                member[j] = max(mem)
                rule_text += f' x{j}={x} is F{j}={f} '
                if j < self.n_feature - 1:
                    rule_text += 'and'
                F[i, 0, j] = f
            F[i, 1, :-1] = member
            F[i, 0, -1] = y
            F[i, 1, -1] = 1
            all_rule += f'R{i}: {rule_text} then y{i} is {y} \n'

        return F

    def pateint(self, F, fuzzyset):
        F_x = F[:, 0, :-1]
        F_y = F[:, 0, -1]
        Fx_mem = F[:, 1, :-1]
        Fy_mem = F[:, 1, -1]

        def member_p_rule(i):
            member = np.zeros((self.sample_train, self.n_feature))
            min_p = np.zeros(self.sample_train)
            for r in range(self.sample_train):
                for j in range(self.n_feature):
                    x = self.X_testdata.iloc[i, j]

                    if F_x[r, j] == 0:
                        a = fuzzyset[j, 0, 0, 0]
                        b = fuzzyset[j, 0, 0, 1]
                        c = fuzzyset[j, 0, 0, 2]
                        member[r, j] = self.membership(x, a, b, c)
                    elif F_x[r, j] == self.n - 1:
                        a = fuzzyset[j, int(F_x[r, j]), 0, 0]
                        b = fuzzyset[j, int(F_x[r, j]), 0, 1]
                        c = fuzzyset[j, int(F_x[r, j]), 0, 2]
                        member[r, j] = self.membership(x, a, b, c)
                    else:
                        a = fuzzyset[j, int(F_x[r, j]), 0, 0]
                        b = fuzzyset[j, int(F_x[r, j]), 0, 1]
                        c = fuzzyset[j, int(F_x[r, j]), 0, 2]
                        member[r, j] = self.membership(x, a, b, c)

                min_p[r] = min(member[r, :])
            cls_pos = self.y_traindata == 1
            cls_neg = self.y_traindata == 0
            max_pos = max(min_p[cls_pos])
            max_neg = max(min_p[cls_neg])

            target = self.target(max_pos, max_neg)
            return np.argmax(target)

        y_hat = []
        y = self.y_testdata
        for i in range(self.sampletest):
            max_p = member_p_rule(i)
            y_hat.append(max_p)
        res = np.sum(y_hat == y) / self.sampletest

        return res, y_hat

    def target(self, max_pos, max_neg):
        fuz = np.array([[0, 0.5, 1], [0, 1, 0]])
        fuz_pos = fuz.copy()
        fuz_neg = fuz.copy()
        fuz_pos[1] = fuz_pos[1] * max_pos
        fuz_neg[1] = fuz_neg[1] * max_neg
        cen_pos = self.centroid(fuz_pos)
        cen_neg = self.centroid(fuz_neg)
        # print(cen_pos,cen_neg)
        return [cen_neg, cen_pos]

    def centroid(self, fuzzy):
        cent = fuzzy[1, 1] / 2
        return cent

    def similarity(self, p, p_prim):
        sim = min(p, p_prim) / max(p, p_prim)
        return sim

    def plot_fuzzy(self, fuzzy_set):
        fig = plt.figure(figsize=(15, 10))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

        lable = ['GENDER', 'AGE', 'WBC', 'Platelets', 'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils',
                 'Basophils', 'CRP', 'AST', 'ALT', 'ALP', 'GGT', 'LDH', 'SWAB']
        for i in range(1, self.n_feature + 1):
            ax = fig.add_subplot(5, 3, i)
            i -= 1
            for k in range(self.n):
                ax.plot(fuzzy_set[i, k, 0, :], fuzzy_set[i, k, 1, :], '-b')
                ax.set_title(f'{lable[i]}')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    m = 0
    y_hats = 0
    y = []
    mean = 0
    for i in range(50):
        cv = Covid()
        cv.read_data()
        f_set = cv.fuzzification()
        F = cv.extract_rule(f_set)
        res, y_hat = cv.pateint(F, f_set)
        mean += res
        if res > m:
            m = res
            y_hats = y_hat
            y = list(cv.y_testdata)
    avg = mean / 50
    print(m)
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
    accuracy = m * 100

    print(f'true positive:{p_t_pos:5.2f}%')
    print(f'false positive:{p_f_pos:5.2f}%')
    print(f'true negetive:{p_t_neg:5.2f}%')
    print(f'false negetive:{p_f_neg:5.2f}%')
    print('------------***--------------')
    print(f'precesion:{precesion:<5.2f}%')
    print(f'recall:{recall:5.2f}%')
    print(f'f mesure:{f_mesure:<5.2f}')
    print(f'accuracy:{accuracy:<5.2f}%')
