import os
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from lime import lime_tabular
import plotly.express as px
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.manifold import MDS

class ModelPerformance:
    def __init__(self, model) -> None:
        self.model=model
        self.train_status=0

    def train_test(self, dataset, test_size = .33, train_set=None, test_set=None, seed=42):
        feat_name = list(set(dataset.columns.values).difference({"target"}))
        X = dataset[feat_name]
        y = dataset["target"]
        
        if train_set==None or test_set==None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
            self.split = [X_train, X_test, y_train, y_test, feat_name]
        else:
            X_train=train_set[feat_name]
            X_test=test_set[feat_name]
            y_train=train_set["target"]
            y_test=test_set["target"]

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)
        self.acc = accuracy_score(pred, y_test)
        self.train_status=1
        return self.acc
    
    def prediction(self, predict_dataset=None, Use_train=True, Use_test=False):
        if self.train_status==0:
            raise InterruptedError("Need to train the model")
        elif Use_train and Use_test:
            raise ValueError("Can't use training set and testing set at the same time")

        if Use_train:
            X = self.split[0]
            y = self.split[2]
        elif Use_test:
            X = self.split[1]
            y = self.split[3]
        else:
            if "target" in predict_dataset.columns.values:
                feat_name = list(set(predict_dataset.columns.values).difference({"target"}))
                X = predict_dataset[feat_name]
                y = predict_dataset["target"]
            else:
                X=predict_dataset

        pred = self.model.predict(X)

        if isinstance(y, pd.Series):
            pred_class = self.get_pred_judge(pred,y.to_numpy())
        else: 
            pred_class=None

        return pred, pred_class

    def get_pred_judge(self, y_pred, y):
        y_pred_class = []
        for i in range(len(y)):
            if y_pred[i]==1 and y[i]==1:
                y_pred_class.append("TP")
            elif y_pred[i]==1 and y[i]==0:
                y_pred_class.append("FP")
            elif y_pred[i]==0 and y[i]==1:
                y_pred_class.append("FN")
            else:
                y_pred_class.append("TN")
        return y_pred_class
    

class EP:
    def __init__(self, dataset) -> None:
        self.dataset=dataset

    def dim_reduct(self, e_profile, method=MDS(), distance="cosine", seed=42, transform = StandardScaler()):
        if transform is not None:
            dis=transform.fit_transform(e_profile)

        if distance=="cosine":
            pca = MDS(n_components=2, random_state=seed, dissimilarity="precomputed")
            dis = 1-pairwise_distances(dis, metric="cosine")
        else:
            pca = MDS(n_components=2, random_state=seed)
        
        return pca.fit_transform((dis))

    def fig_display_2d(self, pca, pca_name, model, e_profile, add_dataset, file_name=None, save=True):
        print(self.dataset)
        df_px_1 = pd.concat((pca, e_profile, pd.DataFrame(add_dataset, index = ["hover name"]).T,pd.DataFrame(np.repeat("att",len(self.dataset)),columns=["data type"])), axis=1)
        df_px_2 = pd.concat((pca, self.dataset, pd.DataFrame(add_dataset, index = ["hover name"]).T, pd.DataFrame(np.repeat("raw",len(self.dataset)),columns=["data type"])), axis=1)
        df_px = pd.concat((df_px_1, df_px_2),axis=0)

        if file_name==None:
            file_name=  f"HD_shap_{model}_{pca_name}_EP"

        fig = px.scatter(df_px.round(4), x="PC1", y="PC2" ,color="hover name", hover_name = "hover name", facet_col="data type",
                        hover_data=self.dataset.columns.values, opacity=0.8, title=file_name,
                    labels={"color": "hover name"})  

        if save==False:             
            fig.show()
        else:
            fig.write_html(f"html/{file_name}.html")

class LimeExplaination(EP):
    def __init__(self, seed=42) -> None:
        super(EP, self).__init__()
        self.seed=seed

    def explain_for_all(self, model, mode = "classification"):
        """

        """
        explainer = lime_tabular.LimeTabularExplainer(self.X, feature_names=self.feat_name, verbose=False, mode=mode, random_state=self.seed)
        profile_lime = dict()
        for j in tqdm(np.arange(len(self.X))):
            if mode == "classification":
                exp = explainer.explain_instance(self.X[j,:], model.predict_proba, num_features = len(self.feat_name))
            else:
                exp = explainer.explain_instance(self.X[j,:], model.predict, num_features = len(self.feat_name))
            exp_prof_lime=dict()
            exp_value = exp.as_map()[1]
            for i in range(len(exp_value)):
                ind_col = exp_value[i][0]
                exp_prof_lime[self.feat_name[ind_col]] = exp_value[i][1]
            profile_lime[j] = exp_prof_lime

        return(pd.DataFrame(profile_lime).T)


class ShapExplanation(EP):
    def __init__(self, dataset, seed=42) -> None:
        super().__init__(dataset)
        self.seed=seed

    def explain_for_all(self, model):
        explainer = shap.Explainer(model,seed=42)
        shap_values = explainer(self.dataset)
        profile_shap = (shap_values.values[...,0])
        return pd.DataFrame(profile_shap)

    # def 

def log(path, file):
    """[Create a log file to record the experiment's logs]
    
    Arguments:
        path {string} -- path to the directory
        file {string} -- file name
    
    Returns:
        [obj] -- [logger that record logs]
    """

    # check if the file exist
    log_file = os.path.join(path, file)

    if not os.path.isfile(log_file):
        open(log_file, "w+").close()

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    
    # create a file handler for output file
    handler = logging.FileHandler(log_file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger