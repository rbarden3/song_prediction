from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle5 as pickle
import numpy as np
#%%
x = []
y = []
with open('all_data.pkl', 'rb') as handle:
    array_df = pickle.load(handle)
# for df in array_df:
#     x.append(df.drop(df.tail(1).index).values.tolist())
#     y.append(df.tail(1).values.tolist()[0])
#%%
for _ in tqdm(range(len(array_df))):
    df = array_df.pop()
    df.dropna(inplace=True)
    if df.shape[0] > 0:
        y_list = df.tail(1).values.tolist()[0]
        x_list = df.drop(df.tail(1).index).values.tolist()
        y.append(y_list)
        x.append(x_list)
        
#%%
print("\nData Formatted...\n")
# df.tail(1).values.tolist()[0]
# padded = pad_sequences(x)
# X = padded
x = pad_sequences(x)
y = np.array(y)
# with open('rnn_x_data.pkl', 'wb') as handle:
#     pickle.dump(x, handle)
# with open('rnn_y_data.pkl', 'wb') as handle:
#     pickle.dump(y, handle)