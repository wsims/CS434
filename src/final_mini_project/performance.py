

class PerformanceEval(object):

    def __init__(self):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.total = 0

    def add_result(self, prediction, label):
        self.total += 1
        if abs(prediction - 1.0) < 1E-10:
            if abs(label - 1.0) < 1E-10:
                self.TP += 1
            else:
                self.FP += 1
        else:
            if abs(label - 1.0) < 1E-10:
                self.FN += 1
            else:
                self.TN += 1

    def __add__(self, other):
        new = PerformanceEval()
        new.TP = self.TP + other.TP
        new.FN = self.FN + other.FN
        new.FP = self.FP + other.FP
        new.TN = self.TN + other.TN
        new.total = self.total + other.total
        return new

    def accuracy(self):
        return float(self.TP + self.TN)/float(self.total)

    def recall(self):
        return float(self.TP)/float(self.TP + self.FN)

    def precision(self):
        return float(self.TP)/float(self.TP + self.FP)

    def F1(self):
        return float(2*self.TP)/float(2*self.TP + self.FP + self.FN)


