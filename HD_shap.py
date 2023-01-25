import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from func import log, ShapExplanation, ModelPerformance
import shap
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

model_name="rf"
usetrain = False
random_seed = 42

# data import and train_test_split

df_HD = pd.read_csv("data/Heart Disease Dataset.csv")
clf = RandomForestClassifier(random_state=random_seed)

## initiate logger
path = "logs/"+datetime.now().strftime("%m_%d")+"/"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)
logger = log(path=path, file=f"{model_name}_shap.logs")
logger.info("-"*15+"Start Session!"+"-"*15)

# model implementation
# accuracy log
clf_eval = ModelPerformance(clf)
acc = clf_eval.train_test(dataset=df_HD)
logger.info(f"Accuracy: {acc}")

# prediction
y_pred, y_pred_class = clf_eval.prediction(df_HD, Use_train=usetrain)
add_data = [y_pred_class]

# initiate explainer using shap
if usetrain==True:
   explain_X = clf_eval.split[0]
   explain_X = pd.DataFrame(explain_X, columns=clf_eval.split[-1])
else:
   explain_X = df_HD[clf_eval.split[-1]]
ep_explanation = ShapExplanation(dataset=explain_X)
ep_profile = ep_explanation.explain_for_all(model=clf)

profile_shap = pd.DataFrame((ep_profile))
profile_shap.columns=clf_eval.split[-1]

## PCA

# profile_shap_pca = pca.fit_transform((cos_dis))
profile_shap_pca = pd.DataFrame(ep_explanation.dim_reduct(e_profile=profile_shap), columns=["PC1","PC2"])

pca_name = "MDS"
model_name = str(clf).split("(")[0]

ep_explanation.fig_display_2d(model=model_name, pca=profile_shap_pca,e_profile=profile_shap, add_dataset = add_data, save=True, pca_name = pca_name) 

## read preprocessed data
X_pca = pd.DataFrame(ep_explanation.dim_reduct(e_profile=explain_X), columns=["PC1","PC2"])
ep_explanation.fig_display_2d(model=model_name, pca=X_pca, e_profile=explain_X, add_dataset = add_data, save=True, pca_name = pca_name, file_name="HD_raw")
