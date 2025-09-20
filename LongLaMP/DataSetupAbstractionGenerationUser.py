import pandas as pd
import numpy as np

"""
Abstract generation is the task of generating scientific paper abstracts based on the title.
Specifically, the user part refers to evaluating users that have not appeared in the training set (a completely different user).
"""

# Train
directory = "~/Desktop/LongLaMP/Abstract_generation_user/train/"
df_abstract_generation_0 = pd.read_parquet(directory + "train-00000-of-00004.parquet")
df_abstract_generation_1 = pd.read_parquet(directory + "train-00001-of-00004.parquet")
df_abstract_generation_2 = pd.read_parquet(directory + "train-00002-of-00004.parquet")
df_abstract_generation_3 = pd.read_parquet(directory + "train-00003-of-00004.parquet")

df_abstract_generation_user_train = pd.concat([df_abstract_generation_0, df_abstract_generation_1, df_abstract_generation_2, df_abstract_generation_3])
print(df_abstract_generation_user_train.head())
print(f"There are {len(df_abstract_generation_user_train)} rows in abstract generation user train dataset.")
print(df_abstract_generation_user_train.columns)

# Val
directory = "~/Desktop/LongLaMP/Abstract_generation_user/val/"
df_abstract_generation_0 = pd.read_parquet(directory + "val-00000-of-00002.parquet")
df_abstract_generation_1 = pd.read_parquet(directory + "val-00001-of-00002.parquet")

df_abstract_generation_user_val = pd.concat([df_abstract_generation_0, df_abstract_generation_1])
print(df_abstract_generation_user_val.head())
print(f"There are {len(df_abstract_generation_user_val)} rows in abstract generation user validation dataset.")
print(df_abstract_generation_user_val.columns)

# Test
directory = "~/Desktop/LongLaMP/Abstract_generation_user/test/"
df_abstract_generation_0 = pd.read_parquet(directory + "test-00000-of-00002.parquet")
df_abstract_generation_1 = pd.read_parquet(directory + "test-00001-of-00002.parquet")

df_abstract_generation_user_test = pd.concat([df_abstract_generation_0, df_abstract_generation_1])
print(df_abstract_generation_user_test.head())
print(f"There are {len(df_abstract_generation_user_test)} rows in abstract generation user testing dataset.")
print(df_abstract_generation_user_test.columns)

"""
Abstract generation user has 4 columns. They are name, input, output, and profile.
name: Name of the author
input: Prompt provided to the LLM. It contains the title of the paper that the abstract is written for
output: The abstract written by the author.
profile: Profile of the author, which contains numerous sets of abstracts previously written by the author. 
"""