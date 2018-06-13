"""Performance Evaluation Module

Contains the PerformanceEval class which is used
to evaluate the performance of a classifier.  This class
allows for recall, precision, F1, and accuracy to be
calculated.

"""
class PerformanceEval(object):
    """Class object used to evaluate classifier performance."""

    def __init__(self):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.total = 0

    def add_result(self, prediction, label):
        """Add a prediction result to the set."""
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
        """Returns the accuracy of all classifications"""
        return float(self.TP + self.TN)/float(self.total)

    def recall(self):
        """Returns the recall of all classifications"""
        return float(self.TP)/float(self.TP + self.FN)

    def precision(self):
        """Returns the precision of all classifications"""
        return float(self.TP)/float(self.TP + self.FP)

    def F1(self):
        """Returns the F1 score of all classifications"""
        return float(2*self.TP)/float(2*self.TP + self.FP + self.FN)


