#%%
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
data=pd.read_csv("C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/고운세상코스메틱/data/data_new_mbti.csv")
data_exist5=data[data['idx_survey5'].isna().transform(lambda x: not x)]
data_nonexist5=data[data['idx_survey5'].isna()]

X=np.array(data_exist5[['answer4','answer5','answer32','answer28','answer29','answer31']])
y=np.array(data_exist5['idx_survey5'])-1
collections.Counter(y)

X = (X - tf.math.reduce_mean(X, axis=0)) / tf.math.reduce_std(X, axis=0) # scaling
K = len(tf.unique(y)[0])                 # number of category
y_true = y
y = tf.keras.utils.to_categorical(y)  # one-hot encoding
y = tf.cast(y, tf.float32)
X = tf.cast(X, tf.float32)
n, p = X.shape
#%%
class BuildModel(tf.keras.models.Model): # 부모 class
    def __init__(self, n, p, K): # initial
        super(BuildModel, self).__init__() # 상속이 이루어지는 부분
        
        self.n = n
        self.p = p
        self.K = K
        
        self.alpha = tf.Variable(tf.random.normal([1,self.K-1], 0, 1), trainable=True)
        self.beta = tf.Variable(tf.random.normal([self.p, 1], 0, 1), trainable=True)
        
    def call(self, X):
        '''reparametrization for ordered condition'''
        theta = []
        theta.append(self.alpha[0, 0])
        for i in range(1, self.K-1):
            theta.append(tf.square(self.alpha[0, i]) + theta[i-1])
        theta = tf.stack(theta)[tf.newaxis, :]
        
        mat1 = tf.nn.sigmoid(theta + tf.matmul(X, self.beta))
        mat1 = tf.concat((mat1, tf.ones((self.n, 1))), axis=-1)
        mat2 = tf.nn.sigmoid(theta + tf.matmul(X, self.beta))
        mat2 = tf.concat((tf.zeros((self.n, 1)), mat2), axis=-1)
        
        return mat1 - mat2

    def predict(self, x):
        theta = []
        theta.append(self.alpha[0, 0])
        for i in range(1, self.K - 1):
            theta.append(tf.square(self.alpha[0, i]) + theta[i - 1])
        theta = tf.stack(theta)[tf.newaxis, :]

        return tf.nn.sigmoid(theta + tf.matmul(x, self.beta))

    def accuracy(self, X, y_true):
        theta = []
        theta.append(self.alpha[0, 0])
        for i in range(1, self.K-1):
            theta.append(tf.square(self.alpha[0, i]) + theta[i-1])
        theta = tf.stack(theta)[tf.newaxis, :]

        mat1 = tf.nn.sigmoid(theta + tf.matmul(X, self.beta))
        mat1 = tf.concat((mat1, tf.ones((self.n, 1))), axis=-1)
        mat2 = tf.nn.sigmoid(theta + tf.matmul(X, self.beta))
        mat2 = tf.concat((tf.zeros((self.n, 1)), mat2), axis=-1)

        y_pred = tf.argmax(mat1 - mat2, axis=1).numpy()

        table = pd.crosstab(y_true, y_pred,rownames=['True'], colnames=['Predicted'], margins=True)
        acc = np.sum(np.diag(table)[:-1]) / self.n
        return table, acc
#%%
iteration = 1000
lr = 0.08

exmodel = BuildModel(n, p, K)
print(exmodel.alpha, exmodel.beta)
optimizer = tf.keras.optimizers.SGD(lr)

for j in tqdm(range(iteration)):
    with tf.GradientTape() as tape:
        result = exmodel(X)
        loss = -tf.reduce_mean(tf.multiply(y, tf.math.log(result + 1e-8)))
        print(loss)
    # update
    grad = tape.gradient(loss, exmodel.trainable_weights)
    optimizer.apply_gradients(zip(grad, exmodel.trainable_weights)) # 1 update
#%%
print(exmodel.alpha, exmodel.beta)
#%%
table, acc = exmodel.accuracy(X, y_true)
print(table)
print(acc)
#%%
#예측
new_x = tf.cast(np.array(data_nonexist5[['answer4','answer5','answer32','answer28','answer29','answer31']]), tf.float32)
new_x = (new_x - tf.math.reduce_mean(new_x, axis=0)) / tf.math.reduce_std(new_x, axis=0) # scaling
pred = exmodel.predict(new_x)
pred_prob =  np.concatenate(([pred.numpy()[:, 0]],
                           [pred.numpy()[:, i+1] - pred.numpy()[:, i] for i in range(K - 1 - 1)] ,
                           [1 - pred.numpy()[:, -1]]),axis=0)

temp_table=pd.DataFrame(pred_prob.T)
temp_table['pred_index5']=[np.argmax(temp_table.iloc[j,:]) for j in range(len(temp_table))]
temp_table['pred_index5']=temp_table['pred_index5']+1


data_nonexist5=pd.concat([data_nonexist5.reset_index(),temp_table['pred_index5'].reset_index()],axis=1)
data_nonexist5=data_nonexist5.iloc[:,3:]

data_nonexist5['idx_survey5'].fillna(data_nonexist5['pred_index5'],inplace=True)
del data_nonexist5['pred_index5']


data_exist5=data_exist5[['num', 's_idx', 's_age', 'sex', 's_sex',
       'stype_do', 'sjumsu_do', 'sresult_do', 'stype_rs', 'sjumsu_rs',
       'sresult_rs', 'stype_np', 'sjumsu_np', 'sresult_np', 'stype_tw',
       'sjumsu_tw', 'sresult_tw', 'jumsu_w', 'jumsu_do', 'jumsu_rs',
       'jumsu_np', 'jumsu_tw', 'slogic', 'slogic1', 'slogic2', 'slogic3',
       'slogic4', 'slogic5', 's1', 's2', 's3', 's4', 's5', 'answer1',
       'answer2', 'answer3', 'answer4', 'answer5', 'answer6', 'answer7',
       'answer8', 'answer9', 'answer10', 'answer11', 'answer12', 'answer13',
       'answer14', 'answer15', 'answer16', 'answer17', 'answer18', 'answer19',
       'answer20', 'answer21', 'answer22', 'answer23', 'answer24', 'answer25',
       'answer26', 'answer27', 'answer28', 'answer29', 'answer30', 'answer31',
       'answer32', 'answer33', 'answer34', 'answer35', 'answer36', 'awre',
       'reg_dt', 'idx', 'num.1', 'idx_survey1', 'idx_survey2', 'idx_survey3',
       'idx_survey4', 'idx_survey5', 'reg_month', 'reg_season', 'spring',
       'summer', 'fall', 'winter', 'sresult_pore', 'sresult_hd', 'jumsu_hd',
       'jumsu_pore', 'sjumsu_hd', 'sjumsu_pore', 'stype_hd', 'new_awre']]
data_nonexist5=data_nonexist5[['num', 's_idx', 's_age', 'sex', 's_sex',
       'stype_do', 'sjumsu_do', 'sresult_do', 'stype_rs', 'sjumsu_rs',
       'sresult_rs', 'stype_np', 'sjumsu_np', 'sresult_np', 'stype_tw',
       'sjumsu_tw', 'sresult_tw', 'jumsu_w', 'jumsu_do', 'jumsu_rs',
       'jumsu_np', 'jumsu_tw', 'slogic', 'slogic1', 'slogic2', 'slogic3',
       'slogic4', 'slogic5', 's1', 's2', 's3', 's4', 's5', 'answer1',
       'answer2', 'answer3', 'answer4', 'answer5', 'answer6', 'answer7',
       'answer8', 'answer9', 'answer10', 'answer11', 'answer12', 'answer13',
       'answer14', 'answer15', 'answer16', 'answer17', 'answer18', 'answer19',
       'answer20', 'answer21', 'answer22', 'answer23', 'answer24', 'answer25',
       'answer26', 'answer27', 'answer28', 'answer29', 'answer30', 'answer31',
       'answer32', 'answer33', 'answer34', 'answer35', 'answer36', 'awre',
       'reg_dt', 'idx', 'num.1', 'idx_survey1', 'idx_survey2', 'idx_survey3',
       'idx_survey4', 'idx_survey5', 'reg_month', 'reg_season', 'spring',
       'summer', 'fall', 'winter', 'sresult_pore', 'sresult_hd', 'jumsu_hd',
       'jumsu_pore', 'sjumsu_hd', 'sjumsu_pore', 'stype_hd', 'new_awre']]
data=pd.concat([data_exist5,data_nonexist5],axis=0)

data.to_csv("C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/고운세상코스메틱/data/data_impute_survey5.csv")
