
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import warnings
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings('ignore')

import PyPluMA
import PyIO
class JointROCCPlugin:
 def input(self, inputfile):
  self.parameters = PyIO.readParameters(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
  stat1_train = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["trainstat"], sep="\t")
  origin_train = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["trainorigin"], sep="\t")
  stat1_test = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["teststat"], sep="\t")
  origin_test = pd.read_csv(PyPluMA.prefix()+"/"+self.parameters["testorigin"], sep="\t")
  model = XGBClassifier()
  model.load_model(PyPluMA.prefix()+"/"+self.parameters["model"])



  # Get predicted probabilities for positive class for training and test sets
  train_y_prob = model.predict_proba(stat1_train)[:, 1]
  test_y_prob = model.predict_proba(stat1_test)[:, 1]

  # Compute ROC curve and ROC area for training set
  train_fpr, train_tpr, _ = roc_curve(origin_train, train_y_prob)
  train_roc_auc = auc(train_fpr, train_tpr)

  # Compute ROC curve and ROC area for test set
  test_fpr, test_tpr, _ = roc_curve(origin_test, test_y_prob)
  test_roc_auc = auc(test_fpr, test_tpr)

  # Plot ROC curve for training set
  plt.figure()
  lw = 2
  plt.plot(train_fpr, train_tpr, color='darkorange', lw=lw, label='Training ROC curve (area = %0.2f)' % train_roc_auc)

  # Plot ROC curve for test set
  plt.plot(test_fpr, test_tpr, color='blue', lw=lw, label='Test ROC curve (area = %0.2f)' % test_roc_auc)

  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic Curve')
  plt.legend(loc="lower right")
  plt.show()
  plt.savefig(outputfile, dpi=1200, bbox_inches='tight')

