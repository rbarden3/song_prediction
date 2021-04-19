def get_col_names(dataframe_df):
    return list(dataframe_df.columns)

def split_x_y(in_df):
    # Col names
    # ['acousticness', 'analysis_url', 'danceability', 'duration_ms', 'energy', 'id', 'instrumentalness',
    # 'key', 'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'track_href', 'type', 'uri', 'valence']
    in_df = in_df.drop(['analysis_url', 'track_href',
                       'uri', 'id', 'type'], axis='columns')
    in_df, last_row = in_df.drop(in_df.tail(1).index), in_df.tail(1)
    x = in_df.mean().tolist()
    y = last_row.values.tolist()[0]

    return {'x': x, 'y': y}


def get_x_y(in_df_arr):
    out = {'x': [], 'y': []}
    # print(in_df_arr)
    for _, val in enumerate(in_df_arr):
        #print("val -> ", val)
        split_data = split_x_y(val)
        out['x'].append(split_data['x'])
        out['y'].append(split_data['y'])
    return out


def split_df_array(arr_df):
    features_arr = []
    for i in range(len(arr_df)):
        features_arr.append(split_x_y(arr_df[i]))

    return features_arr