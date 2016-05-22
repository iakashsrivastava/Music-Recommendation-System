# -*- coding: cp1252 -*-
# user based collaborative filtering
import math;

# unpruned data - 48M triplets
fileName = "train_triplets.txt";

# dictionary for unique indexes for songs
songs = {}

# dictionary for unique indexes for users
users = {}

# dictionary with key = user and value = user's triplet
users_to_triplets = {}

# function which returns a dictionary with key = song and value = count
def songCountMapping(userAttributes):
    songcountpairs = userAttributes.split(',');
    dict ={};
    for songcountpair in songcountpairs:
        if songcountpair != '':
            pair = songcountpair.split(':');
            dict[int(pair[0])] = float(pair[1]);
    
    return dict;
    
# function which returns the average count for a user
def averageRating(userAttributes):
    songcountpairs = userAttributes.split(',');
    length = userAttributes.count(':');
    sum =0.0;
    for songcountpair in songcountpairs:
        if songcountpair != '':
            pair = songcountpair.split(':');
            sum += float(pair[1]);
    
    return sum/length;

# function which returns the list of common songs for two users
def commonSongs(user1Attributes,user2Attributes):
    list1=[];
    list2=[];
    songcountpairs1 = user1Attributes.split(',');
    songcountpairs2 = user2Attributes.split(',');
    
    for songcountpair in songcountpairs1:
        if songcountpair != '':
            pair = songcountpair.split(':');
            list1.append(int(pair[0]));
    
    for songcountpair in songcountpairs2:
        if songcountpair != '':
            pair = songcountpair.split(':');
            list2.append(int(pair[0]));
            
    return list(set(list1).intersection(list2));


print("Mapping each user and song to unique Index and creating a user to triplets dictionary");
with open(fileName,"r") as f:
        
        #counters for user and song unique indexes    
        user_counter = 0;
        song_counter = 0;
        for line in f:
            user,song,count=line.strip().split('\t')
            
            # normalizing count
            count = 1 + math.log(int(count))
            
            if song not in songs:
                songs[song] = song_counter;
                song_counter = song_counter+1;
                
            if user not in users:
                users[user] = user_counter;
                user_counter = user_counter+1;
            
            if str(users[user]) in users_to_triplets:
                triplets_of_user = users_to_triplets[str(users[user])]
                triplets_of_user.append(str(users[user]) + '\t' + str(songs[song]) + '\t' + str(count))
                users_to_triplets[str(users[user])] = triplets_of_user
            else:
                triplets_of_user = []
                triplets_of_user.append(str(users[user]) + '\t' + str(songs[song]) + '\t' + str(count))
                users_to_triplets[str(users[user])] = triplets_of_user
            
#folds
folds = 5
# average MAE
avgMAE = 0.0
# average RMSE
avgRMSE = 0.0
# RMSE
error_sq_sum = 0
# dictionary to hold pruned triplets
users_to_triplets_pruned = {}

for k in range (0, len(users_to_triplets)):
    triplets_of_user = users_to_triplets[str(k)]
    length_of_user_triplets = len(triplets_of_user)
    # pruning criteria
    if length_of_user_triplets < 200:
        continue
    else:
        users_to_triplets_pruned[str(k)] = users_to_triplets[str(k)];
        
#  updating dataset with pruned data
users_to_triplets = users_to_triplets_pruned;

output_file = open('User_based_output.txt','w');
#writing test and train files per fold
for i in range(0, folds):
    print ("Creating Testing and Training file for fold: "+  str(i+1));
    song_to_user_map = {}
    training_file = open('training.txt','w+');
    testing_file = open('testing.txt','w+');
    
    # deciding bucket size for each fold
    for k in users_to_triplets:
        triplets_of_user = users_to_triplets[str(k)]
        length_of_user_triplets = len(triplets_of_user);
        foldSize = length_of_user_triplets / folds
        start = i * foldSize;
        end = start + foldSize -1
        for j in range(0 , length_of_user_triplets):
            # testing
            if (j >= start and j <= end):
                testing_file.write(triplets_of_user[j]+'\n');
            else:
                training_file.write(triplets_of_user[j]+'\n');
    print ("Testing and Training file Created for fold: "+  str(i+1));
    
    training_file.close();
    testing_file.close();
  
    
    print("Creating User Attributes File For Training Data");   
    UserAttributes = open('UserAttributes.txt','w');     
    
    with open("training.txt","r") as file:
        userAttr = '';
        prevUser = '';
        for line in file:
            user, song, count=line.strip().split('\t');
            if(prevUser == user or prevUser == ''):        
                userAttr += song + ':' + count +',';
            else:
                UserAttributes.write(prevUser + '\t' +userAttr+'\n');
                userAttr = song + ':' + count +',';
                
            if song not in song_to_user_map:
                userArray =[]
                userArray.append(user);
                song_to_user_map[song] = userArray;
            else:
                userArray = song_to_user_map[song];
                userArray.append(user);
                song_to_user_map[song] = userArray;
            
            prevUser = user;
            
        UserAttributes.write(prevUser + '\t' +userAttr+'\n');
        UserAttributes.close();
            
    print("Created User Attributes File for Training Data");
    user_to_song_map ={}
    
    print("Loading User Attributes File for Training Data in Dictionary");
    with open("UserAttributes.txt","r") as file1:
        for user_i in file1:
            user1, user1Attr = user_i.strip().split('\t');
            user_to_song_map[user1] = user1Attr;
            
    print("Creating User Attributes File For Testing Data");   
    UserAttributes = open('UserAttributesTesting.txt','w');     
    
    with open("testing.txt","r") as file:
        userAttr = '';
        prevUser = '';
        for line in file:
            user, song, count=line.strip().split('\t');
            if(prevUser == user or prevUser == ''):        
                userAttr += song + ':' + count +',';
            else:
                UserAttributes.write(prevUser + '\t' +userAttr+'\n');
                userAttr = song + ':' + count +',';
                
            prevUser = user;
            
        UserAttributes.write(prevUser + '\t' +userAttr+'\n');
        UserAttributes.close();
            
    print("Created User Attributes File for Testing Data");
    
    print("Loading Testing Data in a Dictionary");   
    testUserAttr ={};
    
    with open("UserAttributesTesting.txt","r") as file:
        for line in file:
                user,attributes=line.strip().split('\t')
                testUserAttr[int(user)] = attributes;
    
    print("Loaded User Attributes File for Training Data in Dictionary");
    
    print("Now calculating similarities and predictions");
    sum = 0; # for MAE
    N = 0; 
    error_sq_sum =0; # for RMSE
    counter = 1
    
    with open("testing.txt","r") as file:
        for line in file:
            N += 1;
            simDic = {}
            counter += 1
            user1, song, count=line.strip().split('\t')
            Ruser1AvgCount=0.0;
            
            if song in song_to_user_map:
                userArray = song_to_user_map[song];
                num =0.0;
                den =0.0;
                for user2 in userArray:
                    user1Attr = user_to_song_map[user1];
                    user2Attr = user_to_song_map[user2];
                    Ruser2AvgCount = averageRating(user2Attr);
                    user2_songcounts = songCountMapping(user2Attr)
                    if (str(user1) + ',' + str(user2)) not in simDic:
                        P = commonSongs(user1Attr,user2Attr);
                        similarity = 0.0;
                        
                        # taking only those users into account for similarities who have atleast
                        # 10 common songs with the test user
                        if(len(P) > 10):
                            user1_songcounts = songCountMapping(user1Attr)
                            Ruser1AvgCount = averageRating(user1Attr);
                            num1 =0.0;
                            user1den = 0.0;
                            user2den = 0.0;
                            for p in P:
                                Ruser1songcount = user1_songcounts[p]
                                Ruser2songcount = user2_songcounts[p]
                                devuser1 =  Ruser1songcount - Ruser1AvgCount
                                devuser2 =  Ruser2songcount - Ruser2AvgCount
                                if devuser1 < 1:
                                    devuser1 = 1
                                if devuser2 < 1:
                                    devuser2 = 1
                                num1 += devuser1* devuser2;
                                user1den += math.pow(devuser1,2);
                                user2den += math.pow(devuser2,2);
                            
                            user1den = math.pow(user1den,0.5);
                            user2den = math.pow(user2den,0.5);
                            
                            deno = user1den * user2den;
                            prevSim = similarity;
                            
                            if(deno != 0 and num1 != 0):
                                similarity = num1/deno;
                                
                            simDic[str(user1) + ',' + str(user2)] = similarity
                            
                            num += similarity * (user2_songcounts[int(song)] - Ruser2AvgCount);
                            den += similarity;
                    else:  
                        similarity = simDic[str(user1) + ',' + str(user2)]
                            
                        num += similarity * (user2_songcounts[int(song)] - Ruser2AvgCount);
                        den += similarity;
                
                prediction = 0.0;
                
                if den !=0:
                    prediction = Ruser1AvgCount +  (num/den);      
                else:
                    prediction = Ruser1AvgCount; 
                    
                if prediction < 0:
                    prediction = 0;
                    
                error = math.fabs(float(count) - prediction) 
                sum += error
                error_sq = math.pow(error, 2)
                error_sq_sum += error_sq
                
                #uncomment this is code to see MAE for every 10000 triplets
                #if counter%10000 == 0:
                #    print str(i)+ ': ' +str(counter) + '  ----->  ' + str(sum/counter) +'\n'
            
                
    foldMAE = sum / N
    avgMAE += foldMAE
    output_file.write('\nMAE for fold '+ str(i) + ' -> ' + str(foldMAE))
    foldRMSE = math.pow(error_sq_sum/N, 0.5)
    avgRMSE += foldRMSE
    output_file.write('\nRMSE for fold '+ str(i) + ' -> ' + str(foldRMSE))

avgMAE = avgMAE / folds
avgRMSE = avgRMSE / folds
output_file.write('\n\nAverage MAE after 5 folds ' + ' -> ' + str(avgMAE))
output_file.write('\nAverage RMSE after 5 folds ' + ' -> ' + str(avgRMSE))
print 'Done'
output_file.close();