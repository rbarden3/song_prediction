#%%
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle5 as pickle
import numpy as np
#%%

with open('all_data.pkl', 'rb') as handle:
    array_df = pickle.load(handle)
# for df in array_df:
#     x.append(df.drop(df.tail(1).index).values.tolist())
#     y.append(df.tail(1).values.tolist()[0])
#%%
def get_model_xy(playlist_count=len(array_df)):
    x = []
    y = []
    for _ in tqdm(range(playlist_count)):
        df = array_df.pop()
        df.dropna(inplace=True)
        if df.shape[0] > 0:
            y_list = df.tail(1).values.tolist()[0]
            x_list = df.drop(df.tail(1).index).values.tolist()
            y.append(y_list)
            x.append(np.asarray(x_list, dtype=np.float32))
    return (x, y)

x, y = get_model_xy(playlist_count=1)
        
#%%
print("\nData Formatted...\n")
# df.tail(1).values.tolist()[0]
# x = pad_sequences(x, value=0)
# X = padded
# x =  X
y = np.array(y)
with open('rnn_x_data_'+str(len(x))+'.pkl', 'wb') as handle:
    pickle.dump(x, handle)
with open('rnn_y_data_'+str(len(x))+'.pkl', 'wb') as handle:
    pickle.dump(y, handle)
# %%
