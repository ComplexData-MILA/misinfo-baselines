import pandas as pd
import glob
import os
for name in glob.glob('data/**/*'):   
    try:
        new_path = name.replace('data','dehydrated_data')
        df = pd.read_json(name, orient='records',lines=True)
        if 'gossipcop' in name or 'politifact' in name:
            df = df[['tweet_id','label']]
        else:
            df = df[['uid','label']]
        if not os.path.isdir('/'.join(new_path.split('/')[:-1])):
            os.makedirs('/'.join(new_path.split('/')[:-1]))
        df.to_json(new_path,orient='records',lines=True)
    except:
        print(name)
        print('/'.join(new_path.split('/')[:-1]))

