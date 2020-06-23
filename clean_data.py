!mkdir .kaggle
!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
!kaggle config set -n path -v{/content}

token = {"username":"jodlyo","key":"1d8e2f917f33a207e3eb82d65a0a66c5"}

import json
import kaggle
import zipfile
import pandas as pd

with open("/Users/joeodonnell-lyons/.kaggle/kaggle.json", "w") as file:
    json.dump(token, file)
!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
!kaggle config set -n path -v{/content}



!kaggle competitions download -c global-wheat-detection -p //Users/joeodonnell-lyons/Desktop/git/wheat_detection




with zipfile.ZipFile('global-wheat-detection.zip', 'r') as zip_ref:
    zip_ref.extractall()
    
df = pd.read_csv('train.csv')


a = df.head(10)
