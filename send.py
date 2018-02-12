import os
import csv
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


def rforest():
    if (True):
        #print ("hello");
        X = []
        y = []
        with open(os.getcwd() + '/train.csv', 'r') as csvfile:
            file = csv.reader(csvfile)
            for count, row in enumerate(file):
                if count == 0:
                    pass
                else:
                    date = re.search("([0-9]{1,2})/([0-9]{1,2})/([0-9]{4})",row[0]).groups()
                    date = [int(x) for x in date]
                    time = re.search("([0-9]{1,2}):([0-9]{1,2})",row[0]).groups()
                    time = [int(x) for x in time]
                    category_string = row[1]
                    dayofweek_string = row[3]
                    pddistrict_string = row[4]
                    longitude = float(row[7])
                    latitude = float(row[8])
                    X_row = date + time + [longitude, latitude,dayofweek_string, pddistrict_string]
                    y_label = category_string
                    X.append(X_row)
                    y.append(y_label)
                    
        dayofweek_set = set()
        pddistrict_set = set()
        for row in X:
            dayofweek_set.add(row[-2])
            pddistrict_set.add(row[-1])
        dayofweek_dict = {item: i for i, item in enumerate(dayofweek_set)}
        #print (dayofweek_dict)
        pddistrict_dict = {item: i for i, item in enumerate(pddistrict_set)}
        #print (pddistrict_dict)
        num_unique_dayofweek = len(dayofweek_dict)
        num_unique_pddistrict = len(pddistrict_dict)
        for i, row in enumerate(X):
            encoded_dayofweek = [0]*num_unique_dayofweek
            encoded_pddistrict = [0]*num_unique_pddistrict
            current_dayofweek = row[-2]
            current_pddistrict = row[-1]
            encoded_dayofweek[dayofweek_dict[current_dayofweek]] = 1
            encoded_pddistrict[pddistrict_dict[current_pddistrict]] = 1
            X[i] = row[:-2] + encoded_dayofweek + encoded_pddistrict
        #print(X)
        # label binarization
        category_set = set()
        for label in y:
            category_set.add(label)
        category_dict = {item: i for i, item in enumerate(sorted(category_set))}
        num_unique_category = len(category_dict)
        for i, label in enumerate(y):
            y[i] = category_dict[label]
    
        #print (y)
        # does CV and fits the best model
        param_grid = {'n_estimators': [20], 'max_features': [3]}
        #print (param_grid)
        pred = RandomForestClassifier(random_state = 50, n_jobs = -1)
        result = GridSearchCV(pred, param_grid = param_grid,refit = True, cv = 4)
        model = result.fit(X, y)
        
       
        return model,num_unique_dayofweek,num_unique_pddistrict,dayofweek_dict,pddistrict_dict,category_set
       
#trained_clf,num_unique_dayofweek,num_unique_pddistrict,dayofweek_dict,pddistrict_dict,category = rforest();
#print (num_unique_dayofweek);