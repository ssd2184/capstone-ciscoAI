import pandas as pd
import numpy as np

"""
Abstract generation is the task of generating scientific paper abstracts based on the title.
Specifically, the temporal part refers to time-based setting. We evaluate whether the model successfully adapted user's writing style based on past posts.
"""

# Train
directory = "~/Desktop/LongLaMP/Abstract_generation_temporal/train/"
df_abstract_generation_0 = pd.read_parquet(directory + "train-00000-of-00007.parquet")
df_abstract_generation_1 = pd.read_parquet(directory + "train-00001-of-00007.parquet")
df_abstract_generation_2 = pd.read_parquet(directory + "train-00002-of-00007.parquet")
df_abstract_generation_3 = pd.read_parquet(directory + "train-00003-of-00007.parquet")
df_abstract_generation_4 = pd.read_parquet(directory + "train-00004-of-00007.parquet")
df_abstract_generation_5 = pd.read_parquet(directory + "train-00005-of-00007.parquet")
df_abstract_generation_6 = pd.read_parquet(directory + "train-00006-of-00007.parquet")

df_abstract_generation_temporal_train = pd.concat([df_abstract_generation_0, df_abstract_generation_1, df_abstract_generation_2, df_abstract_generation_3,
                                                   df_abstract_generation_4, df_abstract_generation_5, df_abstract_generation_6])
print(df_abstract_generation_temporal_train.head())
print(f"There are {len(df_abstract_generation_temporal_train)} rows in abstract generation temporal train dataset.")
print(df_abstract_generation_temporal_train.columns)

# Val
directory = "~/Desktop/LongLaMP/Abstract_generation_temporal/val/"
df_abstract_generation_0 = pd.read_parquet(directory + "val-00000-of-00002.parquet")
df_abstract_generation_1 = pd.read_parquet(directory + "val-00001-of-00002.parquet")

df_abstract_generation_temporal_val = pd.concat([df_abstract_generation_0, df_abstract_generation_1])
print(df_abstract_generation_temporal_val.head())
print(f"There are {len(df_abstract_generation_temporal_val)} rows in abstract generation temporal validation dataset.")
print(df_abstract_generation_temporal_val.columns)

# Test
directory = "~/Desktop/LongLaMP/Abstract_generation_temporal/test/"
df_abstract_generation_0 = pd.read_parquet(directory + "test-00000-of-00002.parquet")
df_abstract_generation_1 = pd.read_parquet(directory + "test-00001-of-00002.parquet")

df_abstract_generation_temporal_test = pd.concat([df_abstract_generation_0, df_abstract_generation_1])
print(df_abstract_generation_temporal_test.head())
print(f"There are {len(df_abstract_generation_temporal_test)} rows in abstract generation temporal testing dataset.")
print(df_abstract_generation_temporal_test.columns)

"""
Abstract generation temporal has 4 columns. They are name, input, output, and profile.
name: Name of the author
input: Prompt provided to the LLM. It contains the title of the paper that the abstract is written for
output: The abstract written by the author.
profile: Profile of the author, which contains numerous sets of abstracts previously written by the author. 
"""