# to be saved in .../Python/Python38/site-packages

import numpy as np
import math

class cm_ratios:
    
    def __init__(self, cm):
        self.cm = cm
        self.accuracy = self.accuracy()
        self.misclassification_rate = self.misclassification_rate()
        self.TPrate = self.TPrate()
        self.TNrate = self.TNrate()
        self.FPrate = self.FPrate()
        self.FNrate = self.FNrate()
        self.precision = self.precision()

    def accuracy(self):
        # Overall, how often is the classifier correct?
        # (tp + tn)/total
        return (self.cm[0,0] + self.cm[1,1])/np.sum(self.cm)

    def misclassification_rate(self):
        # Overall, how often is it wrong?
        # (fp+fn)/total or 1 - Accuracy (aka error rate)
        return (self.cm[0,1] + self.cm[1,0])/np.sum(self.cm)

    def TPrate(self):
        # When it's actually yes, how often does it predict yes?
        # tp/actual_true or 1 - FNrate (aka senstivity/recall)
        return self.cm[1,1]/sum(self.cm[1])

    def precision(self):
        # Ratio between the True Positives and all the Positives.
        # Out of all the points that we predicted True, how many are actually correct?
        return self.cm[1,1]/np.sum(self.cm, axis=0)[1]

    def TNrate(self):
        # When it's actually no, how often does it predict no?
        # tn/actual_false or 1 - FPrate (aka specificity)
        return self.cm[0,0]/sum(self.cm[0])

    def FPrate(self):
        # When it's actually no, how often does it predict yes?
        # fp/actual_false
        return self.cm[0,1]/sum(self.cm[0])

    def FNrate(self):
        # when it's actually yes, how often does it predict no?
        # fn/actual_true 
        return self.cm[1,0]/sum(self.cm[1])


# McFadden's R pseudo square
class McFaddens_R2():
    
    def __init__(self, df, X,y_test,y_pred_prob):
        self.df = df
        self.X = X
        self.y_test = y_test
        self.y_pred_prob = y_pred_prob
        self.LL_fit, self.LL_mean = 0, 0
        self.log_likelihood_fit = []
        
    
    def fit(self):
        # calculating LL_fit
        for i in range(len(self.y_test)):
            if self.y_test[i]:
                temp = math.log(self.y_pred_prob[i])
                self.LL_fit += temp
            else:
                temp = math.log(1 - self.y_pred_prob[i])
                self.LL_fit += temp
            self.log_likelihood_fit.append(temp)
        
        # calculating LL_mean
        overall_probability = self.df.iloc[:, -1].values.tolist().count(1)/self.X.shape[0]
        for i in self.y_test:
            if i:
                self.LL_mean += math.log(overall_probability)
            else:
                self.LL_mean += math.log(1 - overall_probability)
        
        self.R2 = 1 - self.LL_fit / self.LL_mean
        return self.R2

if __name__ == '__main__':
    print('its a module dumbass.')