import numpy as np
import json

# DATA AND LABEL PREPROCESSING
MIN_NUM_FRAMES = 3

with open('alldata.json') as file:
    data = json.load(file)                                            # load data, parse json file, convert into dictionary

datalist = []                                                         # 7248
labellist = []                                                        # 7,428,1 (either 0 or 1), stores mean diff btwn consec data frames
keylist = list(data['data'].keys())                                   # dictionary keys converted into a list

def duplicate(temp):                                                  # creates 3 channels, duplicates 3 times -- C,H,W
    newtemp = np.repeat(temp[np.newaxis, :], 3, axis=0)
    return newtemp

def hundred_to_zero(temp):                                            # replace empty matrix cells (placeholder currently 100) with float 0
    temp[temp == 100] = 0
    return temp

def process_array(temp):
    temp = np.pad(temp, pad_width=((1, 0), (0, 0)), mode='constant')  # originally 8x9, pad to be 9x9
    temp = hundred_to_zero(temp)
    temp = duplicate(temp)                                            # creates 3 channels --> 3x9x9
    temp = temp[np.newaxis, :]
    return temp

for key in keylist:
    if 'L' in data['data'][key].keys():                                # LEFT eye
        if len(data['data'][key]['L']) > MIN_NUM_FRAMES:               # use at least 4 frames, decreases sample size significantly but necessary to filter
            temp0 = np.array(data['data'][key]['L'][0]['td'])          # use 3 temps (~median value)
            temp1 = np.array(data['data'][key]['L'][1]['td'])
            temp2 = np.array(data['data'][key]['L'][2]['td'])
            newtemp0 = process_array(temp0)
            newtemp1 = process_array(temp1)
            newtemp2 = process_array(temp2)
            
            video = np.vstack((newtemp0, newtemp1, newtemp2))          # vertically stack to get 4 channels, add new axis frame channel 3x3x9x9
            datalist.append(video)
            
            total_num_of_frames = len(data['data'][key]['L'])          # calculate mean difference MD between consecutive frames in data
            MD_diff_list = []
            
            for i in range(0, total_num_of_frames - 1):
                age1 = data['data'][key]['L'][i]['age']                # extract two frames
                age2 = data['data'][key]['L'][i+1]['age']
                temp1 = hundred_to_zero(np.array(data['data'][key]['L'][i]['td']))
                temp2 = hundred_to_zero(np.array(data['data'][key]['L'][i+1]['td']))

                MD1 = np.mean(temp1[temp1 != 0])                       # calculate MD using numpy mean function
                MD2 = np.mean(temp2[temp2 != 0])

                MD_diff_per_year = (MD2 - MD1) / (age2 - age1)         # calculate mean difference
                MD_diff_list.append(MD_diff_per_year)

            labellist.append(MD_diff_list)                             # append to MD differences to label list

    if 'R' in data['data'][key].keys():                                # RIGHT eye
        if len(data['data'][key]['R']) > MIN_NUM_FRAMES:
            temp0 = np.array(data['data'][key]['R'][0]['td'])
            temp1 = np.array(data['data'][key]['R'][1]['td'])
            temp2 = np.array(data['data'][key]['R'][2]['td'])
            newtemp0 = process_array(temp0)
            newtemp1 = process_array(temp1)
            newtemp2 = process_array(temp2)
            
            video = np.vstack((newtemp0, newtemp1, newtemp2))
            datalist.append(video)

            total_num_of_frames = len(data['data'][key]['R'])
            MD_diff_list = []
            for i in range(0, total_num_of_frames - 1):
                age1 = data['data'][key]['R'][i]['age']
                age2 = data['data'][key]['R'][i+1]['age']
                temp1 = hundred_to_zero(np.array(data['data'][key]['R'][i]['td']))
                temp2 = hundred_to_zero(np.array(data['data'][key]['R'][i+1]['td']))

                MD1 = np.mean(temp1[temp1 != 0])
                MD2 = np.mean(temp2[temp2 != 0])

                MD_diff_per_year = (MD2 - MD1) / (age2 - age1)
                MD_diff_list.append(MD_diff_per_year)

            labellist.append(MD_diff_list)

prog_labellist = []                                                   # progression list, which labels are progressing
for i in range(len(labellist)):
    if min(labellist[i]) < -1:
        prog_labellist.append(int(1))
    else:
        prog_labellist.append(int(0))

print("Length of prog_labellist, total number of subjects with 3+ frames:", len(prog_labellist))
print("Progressing number of subjects (1):", prog_labellist.count(1))
print("Data preprocessing complete!")