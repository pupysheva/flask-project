import pandas as pd


def get_sorted_df(df, col_name):
    df.sort_values(by=col_name, ascending=False, inplace=True)