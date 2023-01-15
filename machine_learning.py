import nltk
import numpy as np
from tqdm.notebook import tqdm_notebook 
import pandas as pd
import os
import math
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn import preprocessing
import random

import timeit

from ir_system import IRSystem

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import preprocessing

class MachineLearning():
    def load_data(self, path):
        
        
        #_____________ Read data from CISI.ALL file and store in dictinary ________________
        
        with open(os.path.join(path, 'CISI.ALL')) as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")
    
        doc_set = {}
        doc_id = ""
        doc_text = ""

        for l in lines:
            if l.startswith(".I"):
                doc_id = l.split(" ")[1].strip() 
            elif l.startswith(".X"):
                doc_set[doc_id] = doc_text.lstrip(" ")
                doc_id = ""
                doc_text = ""
            elif l.startswith(".T") or l.startswith(".W"):
                doc_text += l.strip()[3:] + " "

        # print(f"Number of documents = {len(doc_set)}")
        # print(doc_set["1"]) 
        
        
        #_____________ Read data from CISI.QRY file and store in dictinary ________________
        
        with open(os.path.join(path, 'CISI.QRY')) as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")
            
        qry_set = {}
        qry_id = ""
        for l in lines:
            if l.startswith(".I"):
                qry_id = l.split(" ")[1].strip() 
            elif l.startswith(".W"):
                qry_set[qry_id] = l.strip()[3:]
                qry_id = ""

        # print(f"\n\nNumber of queries = {len(qry_set)}")    
        # print(qry_set["1"]) 
        
        
        #_____________ Read data from CISI.REL file and store in dictinary ________________
        
        rel_set = {}
        with open(os.path.join(path, 'CISI.REL')) as f:
            for l in f.readlines():
                qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0] 
                doc_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1]

                if qry_id in rel_set:
                    rel_set[qry_id].append(doc_id)
                else:
                    rel_set[qry_id] = []
                    rel_set[qry_id].append(doc_id)

        # print(f"\n\nNumber of mappings = {len(rel_set)}")
        # print(rel_set["1"]) 
        
        doc_set = {int(id):doc for (id,doc) in doc_set.items()}
        qry_set = {int(id):qry for (id,qry) in qry_set.items()}
        rel_set = {int(qid):list(map(int, did_lst)) for (qid,did_lst) in rel_set.items()}
        
        return doc_set, qry_set, rel_set

    
    def __init__(self):
            self.ExpReg = nltk. RegexpTokenizer('(?:[A-Za-z]\.)+|\d+(?:\.\d+)?%?|\w+(?:\-\w+)*')
            self.MotsVides = nltk.corpus.stopwords.words('english')
            self.Porter = nltk.PorterStemmer()
            self.Lancaster = nltk.LancasterStemmer()

            self.df_freqs_poids_porter = pd.read_csv('freq_poids_porter.csv')
            self.df_freqs_poids_lan = pd.read_csv('freq_poids_lancaster.csv')

            self.doc_set, self.qry_set, self.rel_set = self.load_data('documents')


    def produit_scalaire(self,df,query,stemmer = 'P'):
        words = np.unique(self.ExpReg.tokenize(query))
        docs = df.Document.unique()
        if stemmer=='P':
            TermesSansMotsVides = [self.Porter.stem(terme) for terme in words if terme.lower() not in self.MotsVides]
        elif stemmer=='L':
            TermesSansMotsVides = [self.Lancaster.stem(terme) for terme in words if terme.lower() not in self.MotsVides]
        #print(TermesSansMotsVides)
        rows=[]
        for doc in docs:
            result = df[(df['Terme'].isin(TermesSansMotsVides)) & (df['Document']==doc)]
            somme = np.sum(result['Poid'])
            rows.append([doc,somme])
        return TermesSansMotsVides, pd.DataFrame(data=rows,columns=['Document','Poid'])

    def Cosine(self, df,query,stemmer='L'):
        words,produit = self.produit_scalaire(df,query,stemmer)
        taille = len(words)
        rows = []
        docs = df.Document.unique()
        for doc in (docs):
            # temp = df[(df['Terme'].isin(words))&(df['Document']==doc)].assign(square = lambda x:(x['Poid']**2))
            temp = df[df['Document']==doc].assign(square = lambda x:(x['Poid']**2))
            square_root = math.sqrt(np.sum(temp['square']))
            part1 = produit[produit['Document']==doc]['Poid'].values[0]
            resultat = part1/(math.sqrt(taille)*square_root)
            rows.append([doc,resultat])
        return pd.DataFrame(data=rows,columns=['Document','Mesure Cosine']).replace(np.nan,0).sort_values(by='Mesure Cosine',ascending=False).reset_index(drop=True)
        

    def Jaccard(self, df,query,stemmer='L'):
        words,produit = self.produit_scalaire(df,query,stemmer)
        taille = len(words)
        rows = []
        docs = df.Document.unique()
        for doc in tqdm_notebook (docs) :
            # temp = df[(df['Terme'].isin(words))&(df['Document']==doc)].assign(square = lambda x:(x['Poid']**2))
            temp = df[df['Document']==doc].assign(square = lambda x:(x['Poid']**2))
            somme_carres = np.sum(temp['square'])
            # print('somme carres ',somme_carres)
            part1 = produit[produit['Document']==doc]['Poid'].values[0]
            # print('produit :', part1)
            somme_poids = np.sum(df[(df['Terme'].isin(words))&(df['Document']==doc)]['Poid'])
            # print('somme poids : ',somme_poids)
            # print(taille+somme_carres-somme_poids)
            resultat = np.divide(part1,np.add(taille,np.subtract(somme_carres,somme_poids)))
            # resultat = part1/(taille+somme_carres-somme_poids)
            rows.append([doc,resultat])
        return pd.DataFrame(data=rows,columns=['Document','Mesure Jaccard']).replace(np.nan,0).sort_values(by='Mesure Jaccard',ascending=False).reset_index(drop=True)



    def BM25(self, df,query,stemmer='L',K=1.20,B=0.75):
        words = np.unique(self.ExpReg.tokenize(query))
        docs = df.Document.unique()
        if stemmer=='P':
            TermesSansMotsVides = [self.Porter.stem(terme) for terme in words if terme.lower() not in self.MotsVides]
        elif stemmer=='L':
            TermesSansMotsVides = [self.Lancaster.stem(terme) for terme in words if terme.lower() not in self.MotsVides]
        taille = []
        for doc in docs:
            taille.append(len(df[df['Document']==doc]))
        avdl = np.mean(taille)
        N = len(docs)
        nis = pd.DataFrame(df['Terme'].value_counts()).reset_index()
        nis.columns=['Terme','Ni']
        nis=nis[nis['Terme'].isin(TermesSansMotsVides)]
        rows = []    
        for doc in tqdm_notebook(docs):
            somme = 0
            temp = pd.merge(nis,df[df['Document']==doc],on='Terme')
            if(len(temp)==0):
                rows.append([doc,0])
            else: 
                A = np.multiply(K,np.add(np.subtract(1,B),np.multiply(B,np.divide(taille[doc-1],avdl))))        
                temp['RSV'] = temp.apply(lambda x: np.multiply(np.divide(x.Frequence,np.add(A,x.Frequence)),math.log10(np.divide(np.add(np.subtract(N,x.Ni),0.5),np.add(x.Ni,0.5)))),axis=1)
                rows.append([doc,np.sum(temp['RSV'])])

        final = pd.DataFrame(data=rows,columns=['Document','Probabilite BM25']).sort_values(by='Probabilite BM25',ascending=False).reset_index(drop=True)

        return final

    def _distance(self, p1, p2):
            result = 0
            for i in range(len(p1)):
                if(type(p1[i]) == str or type(p2[i]) == str):
                    if(p1[i] != p2[i]):
                        result += 1
                else : result += (p1[i] - p2[i]) ** 2
            return math.sqrt(result)
            #return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
    ## MACHINE LEARNING CLASSES

    class DBSCAN: 
        def __init__(self, eps, min_pts, data):
            self.eps = eps
            self.min_pts = min_pts
            self.data = data
            self.clusters = []
            self.noise = []
            self.core_pts = []
            self.visited = []
            self.clustered = []
            self.cluster_num = 0
            self.clustered_pts = []
            
        def _distance(self, p1, p2):
            result = 0
            for i in range(len(p1)):
                if(type(p1[i]) == str or type(p2[i]) == str):
                    if(p1[i] != p2[i]):
                        result += 1
                else : result += (p1[i] - p2[i]) ** 2
            return math.sqrt(result)
            #return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
        
        def _region_query(self, point):
            neighbors = []
            for i in range(len(self.data)):
                if self._distance(point, self.data[i]) < self.eps:
                    neighbors.append(i)
            return neighbors
        
        def _expand_cluster(self, point, neighbors):
            self.clusters[self.cluster_num].append(point)
            self.clustered.append(point)
            self.visited.append(point)
            for i in neighbors:
                if i not in self.visited:
                    self.visited.append(i)
                    new_neighbors = self._region_query(self.data[i])
                    if len(new_neighbors) >= self.min_pts:
                        neighbors += new_neighbors
                if i not in self.clustered:
                    self.clusters[self.cluster_num].append(i)
                    self.clustered.append(i)
                    
        def fit(self):
            for i in range(len(self.data)):
                if i not in self.visited:
                    self.visited.append(i)
                    neighbors = self._region_query(self.data[i])
                    if len(neighbors) < self.min_pts:
                        self.noise.append(i)
                    else:
                        self.clusters.append([])
                        self._expand_cluster(i, neighbors)
                        self.cluster_num += 1
                        
        def get_clusters(self):
            return self.clusters
        
        def get_noise(self):
            return self.noise


    class NaiveBayesClassifier():
        '''
        Bayes Theorem form
        P(y|X) = P(X|y) * P(y) / P(X)
        '''
        def calc_prior(self, features, target):
            '''
            prior probability P(y)
            calculate prior probabilities
            '''
            self.prior = (features.groupby(target).apply(lambda x: len(x)) / self.rows).to_numpy()

            return self.prior
        
        def calc_statistics(self, features, target):
            '''
            calculate mean, variance for each column and convert to numpy array
            ''' 
            self.mean = features.groupby(target).apply(np.mean).to_numpy()
            self.var = features.groupby(target).apply(np.var).to_numpy()
                
            return self.mean, self.var
        
        def gaussian_density(self, class_idx, x):     
            '''
            calculate probability from gaussian density function (normally distributed)
            we will assume that probability of specific target value given specific class is normally distributed 
            
            probability density function derived from wikipedia:
            (1/√2pi*σ) * exp((-1/2)*((x-μ)^2)/(2*σ²)), where μ is mean, σ² is variance, σ is quare root of variance (standard deviation)
            '''
            mean = self.mean[class_idx]
            var = self.var[class_idx]
            numerator = np.exp((-1/2)*((x-mean)**2) / (2 * var))
    #         numerator = np.exp(-((x-mean)**2 / (2 * var)))
            denominator = np.sqrt(2 * np.pi * var)
            prob = numerator / denominator
            return prob
        
        def calc_posterior(self, x):
            posteriors = []

            # calculate posterior probability for each class
            for i in range(self.count):
                prior = np.log(self.prior[i]) ## use the log to make it more numerically stable
                conditional = np.sum(np.log(self.gaussian_density(i, x))) # use the log to make it more numerically stable
                posterior = prior + conditional
                posteriors.append(posterior)
            # return class with highest posterior probability
            return self.classes[np.argmax(posteriors)]
        

        def fit(self, features, target):
            self.classes = np.unique(target)
            self.count = len(self.classes)
            self.feature_nums = features.shape[1]
            self.rows = features.shape[0]
            
            self.calc_statistics(features, target)
            self.calc_prior(features, target)
            
        def predict(self, features):
            preds = [self.calc_posterior(f) for f in features.to_numpy()]
            return preds

        def accuracy(self, y_test, y_pred):
            accuracy = np.sum(y_test == y_pred) / len(y_test)
            return accuracy
        
        # def confusion_matrix(self, y_true, y_pred):
        #     mask = y_true[y_true == -1.0]
        #     y_true.drop(mask.index, inplace=True)
        #     y_pred.drop(mask.index, inplace=True)
        #     tp = np.sum(y_true * y_pred)
        #     fn = np.sum(y_true * (1 - y_pred))
        #     fp = np.sum((1 - y_true) * y_pred)
        #     tn = np.sum((1 - y_true) * (1 - y_pred))
        #     return np.array([[tp, fn], [fp, tn]])

        #confusion matrix for multiclass
        def confusion_matrix(self, y_true, y_pred):
            cm = np.zeros((self.count, self.count))
            for i in range(len(y_true)):
                cm[int(y_true[i])][int(y_pred[i])] += 1
            return cm

        def precision(self, y_test, y_pred):
            cm = self.confusion_matrix(y_test, y_pred)
            precision = np.diag(cm) / np.sum(cm, axis=0)
            return precision
        
        def recall(self, y_test, y_pred):
            cm = self.confusion_matrix(y_test, y_pred)
            recall = np.diag(cm) / np.sum(cm, axis=1)
            return recall
        
        def f1_score(self, y_test, y_pred):
            precision = self.precision(y_test, y_pred)
            recall = self.recall(y_test, y_pred)
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1

        def visualize(self, y_true, y_pred, target):
            
            tr = pd.DataFrame(data=y_true, columns=[target])
            pr = pd.DataFrame(data=y_pred, columns=[target])
            
            
            fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15,6))
            
            sns.countplot(x=target, data=tr, ax=ax[0], palette='viridis', alpha=0.7, hue=target, dodge=False)
            sns.countplot(x=target, data=pr, ax=ax[1], palette='viridis', alpha=0.7, hue=target, dodge=False)
            

            fig.suptitle('True vs Predicted Comparison', fontsize=20)

            ax[0].tick_params(labelsize=12)
            ax[1].tick_params(labelsize=12)
            ax[0].set_title("True values", fontsize=18)
            ax[1].set_title("Predicted values", fontsize=18)
            # plt.show()
            return fig


    def roc_curve(self, stemmer,query,measure):
        df_freqs_poids_porter= pd.read_csv('freq_poids_porter.csv')
        df_freqs_poids_lan= pd.read_csv('freq_poids_lancaster.csv')
        if measure=='Datamining':
            X_train = pd.read_csv('X_train.csv')
            y_train = pd.read_csv('y_train.csv')
            X_test = pd.read_csv('X_test.csv')
            y_test = pd.read_csv('y_test.csv')
            bayes= self.NaiveBayesClassifier()
            bayes.fit(X_train.squeeze(), y_train.squeeze())
            queries_porter = pd.read_csv('queries_porter.csv')
            pca = PCA(n_components=2)
            pca.fit(queries_porter)
            x_pca = pca.transform(queries_porter)
            pca_df = pd.DataFrame(x_pca)
            pca_df
            x = pca_df.values #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            pca_df_normalized_queries = pd.DataFrame(x_scaled)
            pca_df_normalized_queries['label']=bayes.predict(pca_df_normalized_queries)
            pca_df_dbscan = pd.read_csv('pca_df_dbscan.csv').drop(columns=['Unnamed: 0'])
            docs = pca_df_dbscan[pca_df_dbscan['label'] == pca_df_normalized_queries.loc[query-1]['label']]
            docs['Document']=docs.index+1
            docs['Distance']=np.nan
            cols = ['doc','distance']
            for doc in docs.index+1:
                docs['Distance'].loc[doc-1] = self._distance(pca_df_normalized_queries.loc[query-1,0:1],docs.loc[doc-1][0:2].values)
                    #r = r.append({'doc':doc,'distance':_distance(pca_df_normalized_queries.loc[qry-1,0:1],p.loc[doc-1,0:1])}, ignore_index=True)
            pr = docs.sort_values(by=['Distance'], ascending=True)
            docs = pr.head(10)
            try:
                docs['Pertinent']= docs['Document'].apply(lambda x: 'Oui' if x in self.rel_set[query] else 'Non')
                pr['Pertinent']= pr['Document'].apply(lambda x: 'Oui' if x in self.rel_set[query] else 'Non')
            except:
                docs['Pertinent']= 'Non'
                pr['Pertinent']= 'Non'
            docs['Precision']=0.0
            docs['Rappel']=0.0
            docs['F-Mesure']=0.0
            docs['Precision_Interpolée']=0.0
            docs['Rappel_Interpolée']=0.0
            docs['F-Mesure_Interpolée']=0.0
            c=0
            try:
                p=docs['Pertinent'].value_counts()['Oui']
            except:
                p=0
            
            for i in range(0,10):
                if docs['Pertinent'].iloc[i] == 'Oui':
                    c+=1
                docs['Precision'].iloc[i] = c/(i+1)
                try:
                    docs['Rappel'].iloc[i] = c/p
                except:
                    docs['Rappel'].iloc[i] = 0
                docs['F-Mesure'].iloc[i] = 2*docs['Precision'].iloc[i]*docs['Rappel'].iloc[i]/(docs['Precision'].iloc[i]+docs['Rappel'].iloc[i])
            docs['Precision_Interpolée'] = 0.0
            docs['Rappel_Interpolée'] = 0.0
            docs['F-Mesure_Interpolée'] = 0.0
            k=0
            for i in range(0,10):
                docs['Rappel_Interpolée'].iloc[i]=k/10
                jac= docs[docs['Rappel_Interpolée'].iloc[i] <= docs['Rappel']]
                if jac.empty:
                    docs['Precision_Interpolée'].iloc[i]=0
                else:
                    docs['Precision_Interpolée'].iloc[i]=jac['Precision'].max()
                docs['F-Mesure_Interpolée'].iloc[i] = 2*docs['Precision_Interpolée'].iloc[i]*docs['Rappel_Interpolée'].iloc[i]/(docs['Precision_Interpolée'].iloc[i]+docs['Rappel_Interpolée'].iloc[i])
                k+=1
                matrix = np.zeros((2,2),int)
                matrix[0][0] += p
                try:
                    temp = [doc not in np.array(docs.index+1) for doc in np.array(self.rel_set[query])]
                    matrix[0][1] += temp.count(True)
                except: 
                    matrix[0][1] += 0
                try:
                    temp = [doc not in np.array(self.rel_set[query]) for doc in np.array(pr[(pr['Pertinent']=='Non')].index+1)]
                    matrix[1][0] += temp.count(True)
                except:
                    matrix[1][0] += len(np.array(pr[(pr['Pertinent']=='Non')].index+1))
                try:
                    temp = [doc not in np.array(self.rel_set[query]) for doc in np.array(pca_df_dbscan[pca_df_dbscan['label'] != pca_df_normalized_queries.loc[0]['label']].index+1)]
                    matrix[1][1] += temp.count(True)
                except:
                    matrix[1][1] += len(np.array(pca_df_dbscan[pca_df_dbscan['label'] != pca_df_normalized_queries.loc[0]['label']].index+1))
            return docs , matrix    
        if stemmer=='P':
            if measure=='Produit Scalaire':
                pr = self.produit_scalaire(df_freqs_poids_porter,self.qry_set[query],'P')[1].sort_values(by='Poid',ascending=False).reset_index(drop=True)
                produit = pr.head(10)
                try:
                    produit['Pertinent']= produit['Document'].apply(lambda x: 'Oui' if x in self.rel_set[query] else 'Non')
                except:
                    produit['Pertinent']= 'Non'
                produit['Precision'] = 0.0
                produit['Rappel'] = 0.0
                produit['F-measure'] = 0.0
                c=0
                try:
                    p=produit['Pertinent'].value_counts()['Oui']
                except:
                    p=0
                for i in range (0,10):
                    if produit['Pertinent'].iloc[i] == 'Oui':
                        c+=1
                    produit['Precision'].iloc[i] = c/(i+1)
                    try:
                        produit['Rappel'].iloc[i] = c/p
                    except:
                        produit['Rappel'].iloc[i] = 0
                    produit['F-measure'].iloc[i] = 2*produit['Precision'].iloc[i]*produit['Rappel'].iloc[i]/(produit['Precision'].iloc[i]+produit['Rappel'].iloc[i])
                produit['Precision_Interpolée'] = 0.0
                produit['Rappel_Interpolée'] = 0.0
                produit['F-measure_Interpolée'] = 0.0
                for i in range (0,10):
                    produit['Rappel_Interpolée'].iloc[i]=i/10
                    cos= produit[produit['Rappel_Interpolée'].iloc[i] <= produit['Rappel']]
                    if cos.empty:
                        produit['Precision_Interpolée'].iloc[i]=0
                    else:
                        produit['Precision_Interpolée'].iloc[i]=cos['Precision'].max()
                    produit['F-measure_Interpolée'].iloc[i]=2*produit['Precision_Interpolée'].iloc[i]*produit['Rappel_Interpolée'].iloc[i]/(produit['Precision_Interpolée'].iloc[i]+produit['Rappel_Interpolée'].iloc[i])
                matrix = np.zeros((2,2),int)
                try:
                    l = [a in self.rel_set[query] for a in pr[pr['Poid']!=0]['Document'] ]
                    matrix[0][0] = l.count(True)
                except:
                    matrix[0][0] = 0
                try:
                    l = [a in self.rel_set[query] for a in pr[pr['Poid']==0]['Document'] ]
                    matrix[0][1] = l.count(True)
                except:
                    matrix[0][1] = 0
                try:
                    l = [a not in self.rel_set[query] for a in pr[pr['Poid']!=0]['Document'] ]
                    matrix[1][0] = l.count(True)
                except:
                    matrix[1][0] = len(pr[pr['Poid']!=0]['Document'])
                try:
                    l = [a not in self.rel_set[query] for a in pr[pr['Poid']==0]['Document'] ]
                    matrix[1][1] = l.count(True)
                except:
                    matrix[1][1] = len(pr[pr['Poid']==0]['Document'])
                return produit,matrix
            elif measure == 'Cosine':
                co = self.Cosine(df_freqs_poids_porter,self.qry_set[query],'P')
                cosine = co.copy().head(10)
                try:
                    cosine['Pertinent'] = cosine['Document'].apply(lambda x: 'Oui' if x in self.rel_set[query] else 'Non')
                except:
                    cosine['Pertinent'] = 'Non'
                cosine['Precision'] = 0.0
                cosine['Rappel'] = 0.0
                cosine['F-measure'] = 0.0
                c=0
                try:
                    p=cosine['Pertinent'].value_counts()['Oui']
                except:
                    p=0
                for i in range (0,10):
                    if cosine['Pertinent'].iloc[i] == 'Oui':
                        c+=1
                    cosine['Precision'].iloc[i] = c/(i+1)
                    try:
                        cosine['Rappel'].iloc[i] = c/p
                    except:
                        cosine['Rappel'].iloc[i] = 0
                    cosine['F-measure'].iloc[i] = 2*cosine['Precision'].iloc[i]*cosine['Rappel'].iloc[i]/(cosine['Precision'].iloc[i]+cosine['Rappel'].iloc[i])
                cosine['Precision_Interpolée'] = 0.0
                cosine['Rappel_Interpolée'] = 0.0
                cosine['F-measure_Interpolée'] = 0.0
                for i in range (0,10):
                    cosine['Rappel_Interpolée'].iloc[i]=i/10
                    cos= cosine[cosine['Rappel_Interpolée'].iloc[i] <= cosine['Rappel']]
                    if cos.empty:
                        cosine['Precision_Interpolée'].iloc[i]=0
                    else:
                        cosine['Precision_Interpolée'].iloc[i]=cos['Precision'].max()
                    cosine['F-measure_Interpolée'].iloc[i]=2*cosine['Precision_Interpolée'].iloc[i]*cosine['Rappel_Interpolée'].iloc[i]/(cosine['Precision_Interpolée'].iloc[i]+cosine['Rappel_Interpolée'].iloc[i])
                    matrix = np.zeros((2,2),int)
                    try:
                        l = [a in self.rel_set[query] for a in co[co['Mesure Cosine']!=0]['Document'] ]
                        matrix[0][0] = l.count(True)
                    except:
                        matrix[0][0] = 0
                    try:
                        l = [a in self.rel_set[query] for a in co[co['Mesure Cosine']==0]['Document'] ]
                        matrix[0][1] = l.count(True)
                    except:
                        matrix[0][1] = 0
                    try:
                        l = [a not in self.rel_set[query] for a in co[co['Mesure Cosine']!=0]['Document'] ]
                        matrix[1][0] = l.count(True)
                    except:
                        matrix[1][0] = len(c[c['Mesure Cosine']!=0]['Document'])
                    try:
                        l = [a not in self.rel_set[query] for a in co[co['Mesure Cosine']==0]['Document'] ]
                        matrix[1][1] = l.count(True)
                    except:
                        matrix[1][1] = len(c[c['Mesure Cosine']==0]['Document'])
                return cosine , matrix
            elif measure == 'Jaccard':
                jr = self.Jaccard(df_freqs_poids_porter,self.qry_set[query],'P')
                jaccard = jr.head(10)
                try:
                    jaccard['Pertinent'] = jaccard['Document'].apply(lambda x: 'Oui' if x in self.rel_set[query] else 'Non')
                except:
                    jaccard['Pertinent'] = 'Non'
                jaccard['Precision'] = 0.0
                jaccard['Rappel'] = 0.0
                jaccard['F-measure'] = 0.0
                c=0
                try:
                    p=jaccard['Pertinent'].value_counts()['Oui']
                except:
                    p=0
                for i in range (0,10):
                    if jaccard['Pertinent'].iloc[i] == 'Oui':
                        c+=1
                    jaccard['Precision'].iloc[i] = c/(i+1)
                    try:
                        jaccard['Rappel'].iloc[i] = c/p
                    except:
                        jaccard['Rappel'].iloc[i] = 0
                    jaccard['F-measure'].iloc[i] = 2*jaccard['Precision'].iloc[i]*jaccard['Rappel'].iloc[i]/(jaccard['Precision'].iloc[i]+jaccard['Rappel'].iloc[i])
                jaccard['Precision_Interpolée'] = 0.0
                jaccard['Rappel_Interpolée'] = 0.0
                jaccard['F-measure_Interpolée'] = 0.0
                for i in range (0,10):
                    jaccard['Rappel_Interpolée'].iloc[i]=i/10
                    jac= jaccard[jaccard['Rappel_Interpolée'].iloc[i] <= jaccard['Rappel']]
                    if jac.empty:
                        jaccard['Precision_Interpolée'].iloc[i]=0
                    else:
                        jaccard['Precision_Interpolée'].iloc[i]=jac['Precision'].max()
                    jaccard['F-measure_Interpolée'].iloc[i]=2*jaccard['Precision_Interpolée'].iloc[i]*jaccard['Rappel_Interpolée'].iloc[i]/(jaccard['Precision_Interpolée'].iloc[i]+jaccard['Rappel_Interpolée'].iloc[i])
                matrix = np.zeros((2,2),int)
                try:
                    l = [a in self.rel_set[query] for a in jr[jr['Mesure Jaccard']!=0]['Document'] ]
                    matrix[0][0] = l.count(True)
                except:
                    matrix[0][0] = 0
                try:
                    l = [a in self.rel_set[query] for a in jr[jr['Mesure Jaccard']==0]['Document'] ]
                    matrix[0][1] = l.count(True)
                except:
                    matrix[0][1] = 0
                try:
                    l = [a not in self.rel_set[query] for a in jr[jr['Mesure Jaccard']!=0]['Document'] ]
                    matrix[1][0] = l.count(True)
                except:
                    matrix[1][0] = len(c[c['Mesure Jaccard']!=0]['Document'])
                try:
                    l = [a not in self.rel_set[query] for a in jr[jr['Mesure Jaccard']==0]['Document'] ]
                    matrix[1][1] = l.count(True)
                except:
                    matrix[1][1] = len(jr[jr['Mesure Jaccard']==0]['Document'])
                return jaccard, matrix
            elif measure== 'BM25':
                bmm = self.BM25(df_freqs_poids_porter,self.qry_set[query],'P')
                bm25 = bmm.head(10)
                try:
                    bm25['Pertinent'] = bm25['Document'].apply(lambda x: 'Oui' if x in self.rel_set[query] else 'Non')
                except:
                    bm25['Pertinent'] = 'Non'
                bm25['Precision'] = 0.0
                bm25['Rappel'] = 0.0
                bm25['F-measure'] = 0.0
                c=0
                try:
                    p=bm25['Pertinent'].value_counts()['Oui']
                except:
                    p=0
                for i in range (0,10):
                    if bm25['Pertinent'].iloc[i] == 'Oui':
                        c+=1
                    bm25['Precision'].iloc[i] = c/(i+1)
                    try:
                        bm25['Rappel'].iloc[i] = c/p
                    except:
                        bm25['Rappel'].iloc[i] = 0
                    bm25['F-measure'].iloc[i] = 2*bm25['Precision'].iloc[i]*bm25['Rappel'].iloc[i]/(bm25['Precision'].iloc[i]+bm25['Rappel'].iloc[i])
                bm25['Precision_Interpolée'] = 0.0
                bm25['Rappel_Interpolée'] = 0.0
                bm25['F-measure_Interpolée'] = 0.0
                for i in range (0,10):
                    bm25['Rappel_Interpolée'].iloc[i]=i/10
                    bm= bm25[bm25['Rappel_Interpolée'].iloc[i] <= bm25['Rappel']]
                    if bm.empty:
                        bm25['Precision_Interpolée'].iloc[i]=0
                    else:
                        bm25['Precision_Interpolée'].iloc[i]=bm['Precision'].max()
                    bm25['F-measure_Interpolée'].iloc[i]=2*bm25['Precision_Interpolée'].iloc[i]*bm25['Rappel_Interpolée'].iloc[i]/(bm25['Precision_Interpolée'].iloc[i]+bm25['Rappel_Interpolée'].iloc[i])
                matrix = np.zeros((2,2),int)
                try:
                    l = [a in self.rel_set[query] for a in bmm[bmm['Probabilite BM25']!=0]['Document'] ]
                    matrix[0][0] = l.count(True)
                except:
                    matrix[0][0] = 0
                try:
                    l = [a in self.rel_set[query] for a in bmm[bmm['Probabilite BM25']==0]['Document'] ]
                    matrix[0][1] = l.count(True)
                except:
                    matrix[0][1] = 0
                try:
                    l = [a not in self.rel_set[query] for a in bmm[bmm['Probabilite BM25']!=0]['Document'] ]
                    matrix[1][0] = l.count(True)
                except:
                    matrix[1][0] = len(c[c['Probabilite BM25']!=0]['Document'])
                try:
                    l = [a not in self.rel_set[query] for a in bmm[bmm['Probabilite BM25']==0]['Document'] ]
                    matrix[1][1] = l.count(True)
                except:
                    matrix[1][1] = len(bmm[bmm['Probabilite BM25']==0]['Document'])
                
                return bm25, matrix
            else :
                print('Erreur de mesure')
        elif stemmer=='L':
            if measure=='Produit Scalaire':
                pr = self.produit_scalaire(df_freqs_poids_lan,self.qry_set[query],'L')[1].sort_values(by='Poid',ascending=False).reset_index(drop=True)
                produit = pr.head(10)
                try:
                    produit['Pertinent'] = produit['Document'].apply(lambda x: 'Oui' if x in self.rel_set[query] else 'Non')
                except:
                    produit['Pertinent'] = 'Non'
                produit['Precision'] = 0.0
                produit['Rappel'] = 0.0
                produit['F-measure'] = 0.0
                c=0
                try:
                    p=produit['Pertinent'].value_counts()['Oui']
                except:
                    p=0
                for i in range (0,10):
                    if produit['Pertinent'].iloc[i] == 'Oui':
                        c+=1
                    produit['Precision'].iloc[i] = c/(i+1)
                    produit['Rappel'].iloc[i] = c/p
                    produit['F-measure'].iloc[i] = 2*produit['Precision'].iloc[i]*produit['Rappel'].iloc[i]/(produit['Precision'].iloc[i]+produit['Rappel'].iloc[i])
                produit['Precision_Interpolée'] = 0.0
                produit['Rappel_Interpolée'] = 0.0
                produit['F-measure_Interpolée'] = 0.0
                for i in range (0,10):
                    produit['Rappel_Interpolée'].iloc[i]=i/10
                    cos= produit[produit['Rappel_Interpolée'].iloc[i] <= produit['Rappel']]
                    if cos.empty:
                        produit['Precision_Interpolée'].iloc[i]=0
                    else:
                        produit['Precision_Interpolée'].iloc[i]=cos['Precision'].max()
                    produit['F-measure_Interpolée'].iloc[i]=2*produit['Precision_Interpolée'].iloc[i]*produit['Rappel_Interpolée'].iloc[i]/(produit['Precision_Interpolée'].iloc[i]+produit['Rappel_Interpolée'].iloc[i])
                matrix = np.zeros((2,2),int)
                try:
                    l = [a in self.rel_set[query] for a in pr[pr['Poid']!=0]['Document'] ]
                    matrix[0][0] = l.count(True)
                except:
                    matrix[0][0] = 0
                try:
                    l = [a in self.rel_set[query] for a in pr[pr['Poid']==0]['Document'] ]
                    matrix[0][1] = l.count(True)
                except:
                    matrix[0][1] = 0
                try:
                    l = [a not in self.rel_set[query] for a in pr[pr['Poid']!=0]['Document'] ]
                    matrix[1][0] = l.count(True)
                except:
                    matrix[1][0] = len(pr[pr['Poid']!=0]['Document'])
                try:
                    l = [a not in self.rel_set[query] for a in pr[pr['Poid']==0]['Document'] ]
                    matrix[1][1] = l.count(True)
                except:
                    matrix[1][1] = len(pr[pr['Poid']==0]['Document'])
                return produit, matrix
            elif measure=='Cosine':
                co = self.Cosine(df_freqs_poids_lan,self.qry_set[query],'L')
                cosine = co.head(10)
                try:
                    cosine['Pertinent'] = cosine['Document'].apply(lambda x: 'Oui' if x in self.rel_set[query] else 'Non')
                except:
                    cosine['Pertinent'] = 'Non'
                cosine['Precision'] = 0.0
                cosine['Rappel'] = 0.0
                cosine['F-measure'] = 0.0
                c=0
                try:
                    p=cosine['Pertinent'].value_counts()['Oui']
                except:
                    p=0
                for i in range (0,10):
                    if cosine['Pertinent'].iloc[i] == 'Oui':
                        c+=1
                    cosine['Precision'].iloc[i] = c/(i+1)
                    try:
                        cosine['Rappel'].iloc[i] = c/p
                    except:
                        cosine['Rappel'].iloc[i] = 0
                    cosine['F-measure'].iloc[i] = 2*cosine['Precision'].iloc[i]*cosine['Rappel'].iloc[i]/(cosine['Precision'].iloc[i]+cosine['Rappel'].iloc[i])
                cosine['Precision_Interpolée'] = 0.0
                cosine['Rappel_Interpolée'] = 0.0
                cosine['F-measure_Interpolée'] = 0.0
                for i in range (0,10):
                    cosine['Rappel_Interpolée'].iloc[i]=i/10
                    cos= cosine[cosine['Rappel_Interpolée'].iloc[i] <= cosine['Rappel']]
                    if cos.empty:
                        cosine['Precision_Interpolée'].iloc[i]=0
                    else:
                        cosine['Precision_Interpolée'].iloc[i]=cos['Precision'].max()
                    cosine['F-measure_Interpolée'].iloc[i]=2*cosine['Precision_Interpolée'].iloc[i]*cosine['Rappel_Interpolée'].iloc[i]/(cosine['Precision_Interpolée'].iloc[i]+cosine['Rappel_Interpolée'].iloc[i])
                matrix = np.zeros((2,2),int)
                try:
                    l = [a in self.rel_set[query] for a in co[co['Mesure Cosine']!=0]['Document'] ]
                    matrix[0][0] = l.count(True)
                except:
                    matrix[0][0] = 0
                try:
                    l = [a in self.rel_set[query] for a in co[co['Mesure Cosine']==0]['Document'] ]
                    matrix[0][1] = l.count(True)
                except:
                    matrix[0][1] = 0
                try:
                    l = [a not in self.rel_set[query] for a in co[co['Mesure Cosine']!=0]['Document'] ]
                    matrix[1][0] = l.count(True)
                except:
                    matrix[1][0] = len(c[c['Mesure Cosine']!=0]['Document'])
                try:
                    l = [a not in self.rel_set[query] for a in co[co['Mesure Cosine']==0]['Document'] ]
                    matrix[1][1] = l.count(True)
                except:
                    matrix[1][1] = len(c[c['Mesure Cosine']==0]['Document'])
                return cosine , matrix
            elif measure == 'Jaccard':
                jr = self.Jaccard(df_freqs_poids_lan,self.qry_set[query],'L')
                jaccard = jr.head(10)
                try:
                    jaccard['Pertinent'] = jaccard['Document'].apply(lambda x: 'Oui' if x in self.rel_set[query] else 'Non')
                except:
                    jaccard['Pertinent'] = 'Non'
                jaccard['Precision'] = 0.0
                jaccard['Rappel'] = 0.0
                jaccard['F-measure'] = 0.0
                c=0
                try:
                    p=jaccard['Pertinent'].value_counts()['Oui']
                except:
                    p=0
                for i in range (0,10):
                    if jaccard['Pertinent'].iloc[i] == 'Oui':
                        c+=1
                    jaccard['Precision'].iloc[i] = c/(i+1)
                    try:
                        jaccard['Rappel'].iloc[i] = c/p
                    except:
                        jaccard['Rappel'].iloc[i] = 0
                    jaccard['F-measure'].iloc[i] = 2*jaccard['Precision'].iloc[i]*jaccard['Rappel'].iloc[i]/(jaccard['Precision'].iloc[i]+jaccard['Rappel'].iloc[i])
                jaccard['Precision_Interpolée'] = 0.0
                jaccard['Rappel_Interpolée'] = 0.0
                jaccard['F-measure_Interpolée'] = 0.0
                for i in range (0,10):
                    jaccard['Rappel_Interpolée'][i]=i/10
                    jac= jaccard[jaccard['Rappel_Interpolée'].iloc[i] <= jaccard['Rappel']]
                    if jac.empty:
                        jaccard['Precision_Interpolée'].iloc[i]=0
                    else:
                        jaccard['Precision_Interpolée'].iloc[i]=jac['Precision'].max()
                    jaccard['F-measure_Interpolée'].iloc[i]=2*jaccard['Precision_Interpolée'].iloc[i]*jaccard['Rappel_Interpolée'].iloc[i]/(jaccard['Precision_Interpolée'].iloc[i]+jaccard['Rappel_Interpolée'].iloc[i])
                matrix = np.zeros((2,2),int)
                try:
                    l = [a in self.rel_set[query] for a in jr[jr['Mesure Jaccard']!=0]['Document'] ]
                    matrix[0][0] = l.count(True)
                except:
                    matrix[0][0] = 0
                try:
                    l = [a in self.rel_set[query] for a in jr[jr['Mesure Jaccard']==0]['Document'] ]
                    matrix[0][1] = l.count(True)
                except:
                    matrix[0][1] = 0
                try:
                    l = [a not in self.rel_set[query] for a in jr[jr['Mesure Jaccard']!=0]['Document'] ]
                    matrix[1][0] = l.count(True)
                except:
                    matrix[1][0] = len(c[c['Mesure Jaccard']!=0]['Document'])
                try:
                    l = [a not in self.rel_set[query] for a in jr[jr['Mesure Jaccard']==0]['Document'] ]
                    matrix[1][1] = l.count(True)
                except:
                    matrix[1][1] = len(jr[jr['Mesure Jaccard']==0]['Document'])
                return jaccard, matrix
            elif measure== 'BM25':
                bmm = self.BM25(df_freqs_poids_lan,self.qry_set[query],'L')
                bm25 = bmm.head(10)
                try:
                    bm25['Pertinent'] = bm25['Document'].apply(lambda x: 'Oui' if x in self.rel_set[query] else 'Non')
                except:
                    bm25['Pertinent'] = 'Non'
                bm25['Precision'] = 0.0
                bm25['Rappel'] = 0.0
                bm25['F-measure'] = 0.0
                c=0
                try:
                    p=bm25['Pertinent'].value_counts()['Oui']
                except:
                    p=0
                for i in range (0,10):
                    if bm25['Pertinent'].iloc[i] == 'Oui':
                        c+=1
                    bm25['Precision'].iloc[i] = c/(i+1)
                    try:
                        bm25['Rappel'].iloc[i] = c/p
                    except:
                        bm25['Rappel'].iloc[i] = 0
                    bm25['F-measure'].iloc[i] = 2*bm25['Precision'].iloc[i]*bm25['Rappel'].iloc[i]/(bm25['Precision'].iloc[i]+bm25['Rappel'].iloc[i])
                bm25['Precision_Interpolée'] = 0.0
                bm25['Rappel_Interpolée'] = 0.0
                bm25['F-measure_Interpolée'] = 0.0
                for i in range (0,10):
                    bm25['Rappel_Interpolée'].iloc[i]=i/10
                    bm= bm25[bm25['Rappel_Interpolée'].iloc[i] <= bm25['Rappel']]
                    if bm.empty:
                        bm25['Precision_Interpolée'].iloc[i]=0
                    else:
                        bm25['Precision_Interpolée'].iloc[i]=bm['Precision'].max()
                    bm25['F-measure_Interpolée'].iloc[i]=2*bm25['Precision_Interpolée'].iloc[i]*bm25['Rappel_Interpolée'].iloc[i]/(bm25['Precision_Interpolée'].iloc[i]+bm25['Rappel_Interpolée'].iloc[i])
                matrix = np.zeros((2,2),int)
                try:
                    l = [a in self.rel_set[query] for a in bmm[bmm['Probabilite BM25']!=0]['Document'] ]
                    matrix[0][0] = l.count(True)
                except:
                    matrix[0][0] = 0
                try:
                    l = [a in self.rel_set[query] for a in bmm[bmm['Probabilite BM25']==0]['Document'] ]
                    matrix[0][1] = l.count(True)
                except:
                    matrix[0][1] = 0
                try:
                    l = [a not in self.rel_set[query] for a in bmm[bmm['Probabilite BM25']!=0]['Document'] ]
                    matrix[1][0] = l.count(True)
                except:
                    matrix[1][0] = len(c[c['Probabilite BM25']!=0]['Document'])
                try:
                    l = [a not in self.rel_set[query] for a in bmm[bmm['Probabilite BM25']==0]['Document'] ]
                    matrix[1][1] = l.count(True)
                except:
                    matrix[1][1] = len(bmm[bmm['Probabilite BM25']==0]['Document'])
                
                return bm25, matrix
            else : 
                print('Erreur de mesure')
        else : 
            print('Erreur de stemmer')
        
    def train_test_split(self, df, test_size):
            # make sure it is a float and get the number of instances in the test set
            if isinstance(test_size, float):
                    test_size = round(test_size * len(df))
            # get the indices for the test set
            indices = df.index.tolist()
            # choose them randomly
            test_indices = random.sample(population=indices, k=test_size)
            # separate into test and train
            test_df = df.loc[test_indices]
            train_df = df.drop(test_indices)
            
            return train_df.iloc[:,:-1], test_df.iloc[:,:-1], train_df.iloc[:,-1], test_df.iloc[:,-1]

    def boolean(self, query,stemmer):
        stop_words = ['is', 'a', 'for', 'the', 'of']
        # args = parse_args()
        ir = IRSystem(self.doc_set, stop_words=stop_words)    
        start = timeit.default_timer()
        results = ir.process_query(query)
        stop = timeit.default_timer()
        if results is not None:
            print ('Processing time: {:.5} secs'.format(stop - start))
            # print('\nDoc IDS: ')
            # print(results)    
            if stemmer == 'P':
                q = self.ExpReg.tokenize(query)
                q = [self.Porter.stem(terme) for terme in q if terme.lower() not in self.MotsVides]
                return self.df_freqs_poids_porter[(self.df_freqs_poids_porter['Terme'].isin(q)) & (self.df_freqs_poids_porter['Document'].isin(results))]
            elif stemmer == 'L':
                q = self.ExpReg.tokenize(query)
                q = [self.Lancaster.stem(terme) for terme in q if terme.lower() not in self.MotsVides]
                return self.df_freqs_poids_lan[(self.df_freqs_poids_lan['Terme'].isin(q)) & (self.df_freqs_poids_lan['Document'].isin(results))]

    def freq_inverse(self, df,query,stemmer):
        words = np.unique(self.ExpReg.tokenize(query))
        docs = df.Document.unique()
        if stemmer=='P':
            TermesSansMotsVides = [self.Porter.stem(terme) for terme in words if terme.lower() not in self.MotsVides]
        elif stemmer=='L':
            TermesSansMotsVides = [self.Lancaster.stem(terme) for terme in words if terme.lower() not in self.MotsVides]
        results = df[(df['Terme'].isin(TermesSansMotsVides)) & (df['Frequence'] != 0)]
        #results = results.drop('Terme',axis=1).reset_index(drop=True)
        return results
