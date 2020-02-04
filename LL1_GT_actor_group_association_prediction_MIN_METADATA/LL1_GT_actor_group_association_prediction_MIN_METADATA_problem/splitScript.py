import os, sys
import pandas as pd 

ldfpath = sys.argv[1] # path to learningData.csv
dsfpath = sys.argv[2] # path to dataSplits.csv
assert os.path.exists(ldfpath)

#load dataframe
ldf = pd.read_csv(ldfpath)

#index test samples. assign type
test_idx=(ldf[ldf['timestep']>=4000]).index
ldf['type'] = ['TRAIN']*len(ldf)
ldf.loc[test_idx, 'type'] = 'TEST'

print(ldf[ldf['type']=='TRAIN'].shape, ldf[ldf['type']=='TEST'].shape)

#assign fold and repeat
ldf['fold']=[0]*len(ldf)
ldf['repeat']=[0]*len(ldf)

#remove other columns
ldf = ldf[['d3mIndex','type','fold','repeat']]
ldf = ldf.set_index('d3mIndex')

print(ldf.head())
print(ldf.tail())

#save dataSplits
ldf.to_csv(dsfpath)
