'''Author: Yang Shi yshi26@ncsu.edu'''
import math
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

main_df = pd.read_csv('../../../dataset/CodeWorkout/MainTable.csv')

main_df = main_df[main_df["EventType"] == "Run.Program"]
main_df = main_df.astype({"AssignmentID":int})

# sort_by_time = False    # original Code-DKT experiment
sort_by_time = True     

assignment_list = sorted(list(main_df["AssignmentID"].unique()))
for assignment_id in assignment_list:
    assignment_df = main_df[main_df["AssignmentID"] == assignment_id]

    # test 용. 2026.2.25
    # 만약 정렬을 안하면 DKT가 0.0005로 학습되었을때 CodeDKT보다 더 높다. 그런데, 정렬을 하면 높지는 않다.
    # 이것은, CodeDKT가 원래 그럭저럭 성능에 아무 쓸모 없다는 것이 아니라, 정렬을 하지 않고 데이터셋을 사용했을 때 순서에 따라 data leakage 가 발생한다고 볼수 있다.
    # 첫번째 assignment 만 정렬했을 때와 안했을 때를 문제 ID와 score 만 저장해보자!!!

    # 'col1'과 'col3' 컬럼만 선택해서 저장
    assignment_df.sort_values(by=['SubjectID']).to_csv(f'../data/{assignment_id}_not_sorted.csv', columns=['SubjectID', 'ProblemID', 'Score', 'ServerTimestamp'], index=False)
    assignment_df.sort_values(by=['SubjectID', 'ServerTimestamp']).to_csv(f'../data/{assignment_id}_sorted.csv', columns=['SubjectID', 'ProblemID', 'Score', 'ServerTimestamp'], index=False)

    if sort_by_time:
        save_dir = os.path.join("../data_sort_by_time", str(assignment_id))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = os.path.join("../data", str(assignment_id))

    students = pd.unique(assignment_df["SubjectID"])

    problems = pd.unique(assignment_df["ProblemID"])
    problems_d = {k:v for (v,k) in enumerate(problems) }

    d = {}
    for s in students:
        d[s] = {}
        df = assignment_df[assignment_df["SubjectID"] == s]
        if sort_by_time:
            df = df.sort_values(by=['ServerTimestamp'])

        d[s]["length"] = len(df)
        d[s]["Problems"] = [str(problems_d[i]) for i in df["ProblemID"]]
        d[s]["Result"] = list((df["Score"]==1).astype(int).astype(str))
        d[s]["CodeStates"] = [str(x) for x in list(df["CodeStateID"])]
        
    train_val_s, test_s = train_test_split(students, test_size=0.2, random_state=1)

    np.save(os.path.join(save_dir, "training_students.npy"), train_val_s)
    np.save(os.path.join(save_dir, "testing_students.npy"), test_s)

    if not os.path.isdir(os.path.join(save_dir, "DKTFeatures")):
        os.mkdir(os.path.join(save_dir, "DKTFeatures"))

    file_test = open(os.path.join(save_dir, "DKTFeatures/test_data.csv"),"w")
    for s in test_s:
        if d[s]['length']>0:
            file_test.write(str(d[s]['length']))
            file_test.write(",\n")
            file_test.write(",".join(d[s]['CodeStates']))
            file_test.write(",\n")
            file_test.write(",".join(d[s]['Problems']))
            file_test.write(",\n")
            file_test.write(",".join(d[s]['Result']))
            file_test.write(",\n")
            
    for fold in range(100):
        train_s, val_s = train_test_split(train_val_s, test_size=0.25, random_state=fold)

        file_train = open(os.path.join(save_dir, "DKTFeatures/train_firstatt_"+str(fold)+".csv"),"w")
        for s in train_s:
            if d[s]['length']>0:
                file_train.write(str(d[s]['length']))
                file_train.write(",\n")
                file_train.write(",".join(d[s]['CodeStates']))
                file_train.write(",\n")
                file_train.write(",".join(d[s]['Problems']))
                file_train.write(",\n")
                file_train.write(",".join(d[s]['Result']))
                file_train.write(",\n")


        file_val = open(os.path.join(save_dir, "DKTFeatures/val_firstatt_"+str(fold)+".csv"),"w")
        for s in val_s:
            if d[s]['length']>0:
                file_val.write(str(d[s]['length']))
                file_val.write(",\n")
                file_val.write(",".join(d[s]['CodeStates']))
                file_val.write(",\n")
                file_val.write(",".join(d[s]['Problems']))
                file_val.write(",\n")
                file_val.write(",".join(d[s]['Result']))
                file_val.write(",\n")