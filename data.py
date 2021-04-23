def get_col_names(dataframe_df):
    return list(dataframe_df.columns)

def split_x_y(in_df, average=True):
    # Col names
    # ['acousticness', 'analysis_url', 'danceability', 'duration_ms', 'energy', 'id', 'instrumentalness',
    # 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'track_href', 'type', 'uri', 'valence']
    x = []
    y = []
    in_df = in_df.drop(['analysis_url', 'track_href',
                       'uri', 'id', 'type'], axis='columns')
    if average:
        while in_df.shape[0] > 1:
            in_df, last_row = in_df.drop(in_df.tail(1).index), in_df.tail(1)
            x.append(in_df.mean().tolist())
            y.append(last_row.values.tolist()[0])
    else:
        while in_df.shape[0] > 1:
            in_df, last_row = in_df.drop(in_df.tail(1).index), in_df.tail(1)
            x.append(in_df.values.tolist())
            y.append(last_row.values.tolist()[0])

    return {'x': x, 'y': y}


def get_x_y(in_df_arr, average=True):
    out = {'x': [], 'y': []}
    # print(in_df_arr)
    for _, val in enumerate(in_df_arr):
        #print("val -> ", val)
        split_data = split_x_y(val, average)
        for ind, _ in enumerate(split_data['x']):
            out['x'].append(split_data['x'][ind])
            out['y'].append(split_data['y'][ind])
    return out


def split_df_array(arr_df):
    features_arr = []
    for i in range(len(arr_df)):
        features_arr.append(split_x_y(arr_df[i]))

    return features_arr