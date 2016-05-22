# -*- coding: cp1252 -*-
# user based collaborative filtering
import math

# unpruned data - 48M triplets
fileName = "train_triplets.txt";

# dictionary for unique indexes for songs
songs = {}

# dictionary for unique indexes for users
users = {}

# dictionary with key = user and value = user's triplet
users_to_triplets = {}

# dictionary with key = song and value = song's triplet
songs_to_triplets = {}

# function which returns a dictionary with key = user and value = count
def userCountMapping(UserAttributes):
    usercountpairs = UserAttributes.split(',');
    dict ={};
    for usercountpair in usercountpairs:
        if usercountpair != '':
            pair = usercountpair.split(':');
            dict[int(pair[0])] = float(pair[1]);
    
    return dict;

# function which returns the average count for a song
def averageRating(UserAttributes):
    usercountpairs = UserAttributes.split(',');
    length = UserAttributes.count(':');
    sum =0.0;
    for usercountpair in usercountpairs:
        if usercountpair != '':
            pair = usercountpair.split(':');
            sum += float(pair[1]);
    
    return sum/length;

# function which returns the list of common users for two songs
def commonUsers(song1Attributes,song2Attributes):
    list1=[];
    list2=[];
    usercountpairs1 = song1Attributes.split(',');
    usercountpairs2 = song2Attributes.split(',');
    
    for usercountpair in usercountpairs1:
        if usercountpair != '':
            pair = usercountpair.split(':');
            list1.append(int(pair[0]));
    
    for usercountpair in usercountpairs2:
        if usercountpair != '':
            pair = usercountpair.split(':');
            list2.append(int(pair[0]));
            
    return list(set(list1).intersection(list2));

print("Mapping each user and song to unique Index and creating a song to triplets dictionary");
with open(fileName,"r") as f:
    
        #counters for user and song unique indexes  
        user_counter=0;
        song_counter=0;
        
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


print ("Creating a pruned file")
pruned_file = open('PrunedFileUserBased.txt','w');                    
for k in range (0, len(users_to_triplets)):
    triplets_of_user = users_to_triplets[str(k)]
    length_of_user_triplets = len(triplets_of_user)
    
    # pruning criteria
    if length_of_user_triplets < 200:
        continue
        
    for entry in triplets_of_user:
        user, song, count = entry.split('\t')
        pruned_file.write(str(song) + '\t' + str(user) + '\t' + str(count) + '\n')
print ("Created pruned file")
pruned_file.close()   

with open('PrunedFileSongBased.txt',"r") as f:
    for line in f:
        song,user,count=line.strip().split('\t')
        if song in songs_to_triplets:
            triplets_of_song = songs_to_triplets[song]
            triplets_of_song.append(song + '\t' + user + '\t' + count)
            songs_to_triplets[song] = triplets_of_song
        else:
            triplets_of_song = []
            triplets_of_song.append(song + '\t' + user + '\t' + count)
            songs_to_triplets[song] = triplets_of_song
    
                                                                    
#folds
folds = 5
# average MAE
avgMAE = 0.0
# average RMSE
avgRMSE = 0.0
# RMSE
error_sq_sum = 0

output_file = open('Song_based_output.txt','w');
#writing test and train files per fold
for i in range(0, 5):
    print ("Creating Testing and Training file for fold: "+  str(i+1));
    user_songMapping={}
    training_file = open('training.txt','w+');
    testing_file = open('testing.txt','w+');
    
    # deciding bucket size for each fold
    for k in songs_to_triplets:
        triplets_of_song = songs_to_triplets[k]
        length_of_song_triplets = len(triplets_of_song);
        foldSize = length_of_song_triplets / folds
        start = i * foldSize;
        end = start + foldSize -1
        for j in range(0 , length_of_song_triplets):
            # testing
            if (j >= start and j <= end):
                testing_file.write(triplets_of_song[j]+'\n');
            else:
                training_file.write(triplets_of_song[j]+'\n');
    print ("Testing and Training file Created for fold: "+  str(i+1));
    
    training_file.close();
    testing_file.close();
 
    print("Creating Song Attributes File For Training Data");   
    SongAttributes = open('SongAttributes.txt','w');     
    
    with open("training.txt","r") as file:
        songAttr='';
        prevSong ='';
        for line in file:
            song,user,count=line.strip().split('\t');
            if(prevSong == song or prevSong == ''):        
                songAttr += user + ':' + count +',';
            else:
                SongAttributes.write(prevSong + '\t' +songAttr+'\n');
                songAttr = user + ':' + count +',';
                
            if user not in user_songMapping:
                songArray =[]
                songArray.append(song);
                user_songMapping[user] = songArray;
            else:
                songArray = user_songMapping[user];
                songArray.append(song);
                user_songMapping[user] = songArray;
            
            prevSong = song;
        SongAttributes.write(prevSong + '\t' +songAttr+'\n');
        SongAttributes.close();
            
    print("Created Song Attributes File for Training Data");
    song_userMap ={}
    
    print("Loading Song Attributes File for Training Data in Dictionary");
    with open("SongAttributes.txt","r") as file1:
        for song_i in file1:
            song1, song1Attr = song_i.strip().split('\t');
            song_userMap[song1] = song1Attr;
            
    print("Creating Song Attributes File For Testing Data");   
    SongAttributes = open('SongAttributesTesting.txt','w');     
    
    with open("testing.txt","r") as file:
        songAttr='';
        prevSong ='';
        for line in file:
            user,song,count=line.strip().split('\t');
            if(prevSong == user or prevSong == ''):        
                songAttr += song + ':' + count +',';
            else:
                SongAttributes.write(prevSong + '\t' +songAttr+'\n');
                songAttr = song + ':' + count +',';
                
            prevSong = user;
        SongAttributes.write(prevSong + '\t' +songAttr+'\n');
        SongAttributes.close();
            
    print("Created Song Attributes File for Testing Data");
 
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
            if counter > 50:
                break
            song1,user1,count=line.strip().split('\t')
            Rsong1AvgCount=0.0;
            if user1 in user_songMapping:
                songArray = user_songMapping[user1];
                num = 0.0;
                den = 0.0;
                for song2 in songArray:
                    song1Attr = song_userMap[song1];
                    song2Attr = song_userMap[song2];
                    Rsong2AvgCount = averageRating(song2Attr);
                    song2_usercounts = userCountMapping(song2Attr)
                    if (str(song1) + ',' + str(song2)) not in simDic:
                        P = commonUsers(song1Attr,song2Attr);
                        similarity = 0.0;
                        
                        if(len(P) > 0):
                            song1_usercounts = userCountMapping(song1Attr)
                            Rsong1AvgCount = averageRating(song1Attr);
                            num1 = 0.0;
                            song1den = 0.0;
                            song2den = 0.0;
                            for p in P:
                                Rsong1usercount = song1_usercounts[p]
                                Rsong2usercount = song2_usercounts[p]
                                devsong1 =  Rsong1usercount - Rsong1AvgCount
                                devsong2 =  Rsong2usercount - Rsong2AvgCount
                                if devsong1 < 1:
                                    devsong1 = 1
                                if devsong2 < 1:
                                    devsong2 = 1
                                num1 += devsong1 * devsong2;
                                song1den += math.pow(devsong1,2);
                                song2den += math.pow(devsong2,2);
                            
                            song1den = math.pow(song1den,0.5);
                            song2den = math.pow(song2den,0.5);
                            deno = song1den * song2den;
                            prevSim = similarity;
                            if(deno !=0 and num1!=0):
                                similarity = num1/deno;
                                
                            simDic[str(song1) + ',' + str(song2)] = similarity
                            
                            num += similarity * (song2_usercounts[int(user1)] - Rsong2AvgCount);
                            den += similarity;
                    else: 
                        similarity = simDic[str(song1) + ',' + str(song2)]
                            
                        num += similarity * (song2_usercounts[int(user1)] - Rsong2AvgCount);
                        den += similarity;
                
                prediction =0.0;
                
                if den !=0:
                    prediction = Rsong1AvgCount +  (num/den);      
                else:
                    prediction = Rsong1AvgCount; 
                    
                if prediction <0:
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