import CF_model, math
import numpy as np
#import ModelBased


components = 10
pruning = True # True for pruned dataset, False for the whole dataset

tr = ""

eta=0.01
lamd=0.05
count = 0
MAX_ITER=200
user_counter=0;
song_counter=0;
n_triplets = 0;
dataset = ""
if pruning:
    tr = 'train_triplets_concise.txt'    
    n_triplets = 7858609 #unpruned: 48373586
    user_counter=26386  #unpruned: 1019318 
    song_counter=297053  #unpruned: 384546
    dataset = "pruned"
else:
    tr = 'train_triplets_new.txt'    
    n_triplets = 48373586
    user_counter= 1019318 
    song_counter= 384546
    dataset = "whole"
    
    
print("For %s dataset number of triplets is %d, users %d and songs %d"%(dataset,n_triplets,user_counter,song_counter))
X = np.empty((n_triplets,3,))
count = 0
with open(tr,"r") as f:
        for line in f:
            user, item, rating = line.strip().split('\t')
            userId = int(user)
            itemId = int(item)
            rate = 1+math.log(int(rating))
            X[count,:] = np.array([userId,itemId,rate])
            count += 1

mae = 0.0
rmse = 0.0
k = 5
binSize = n_triplets / k
indexes = range(n_triplets)
np.random.shuffle(indexes)
Xs = X[indexes,:]
del X
for  i in range(k):
    if i == 0 :
        trainSet = Xs[binSize:,:]
        testSet = Xs[0:binSize,:]
    elif i == k - 1 :
        trainSet = Xs[0:(k - 1) * binSize,:]
        testSet = Xs[(k - 1) * binSize:,:]
    else:
        testSet = Xs[i * binSize: (i + 1) * binSize,:]
        tr1 = Xs[:i * binSize,:]
        tr2 = Xs[(i + 1) * binSize:,:]
        trainSet = np.vstack([tr1,tr2])


    cf = CF_model.CFModel(n_items=song_counter, n_users=user_counter, n_components=components)
    Rui_tr = cf.createMap(trainSet)
    Rui_te = cf.createMap(testSet)
    cf.run(Rui_tr)
    temp = cf.eval_MAE(Rui_te)
    print("Round %d: MAE = %f"%(i,temp))

    mae += temp

    temp = cf.eval_RMSE(Rui_te)
    print("Round %d: RMSE = %f"%(i,temp))
    rmse += temp

print("The average MAE = %f"%(mae/k))
print("The average RMSE = %f"%(rmse/k))
