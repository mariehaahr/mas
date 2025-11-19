
import pandas as pd 


df1 = pd.read_csv('/home/rp-fril-mhpe/results-sarc-ratio.csv')
df2 = pd.read_csv('/home/rp-fril-mhpe/first-results-sarc-ratio.csv')


df1_norm = df1.copy()
df2_norm = df2.copy()

# Replace NaN with 0 only in valid_json column
df1_norm['valid_json'] = df1_norm['valid_json'].fillna(0)
df2_norm['valid_json'] = df2_norm['valid_json'].fillna(0)

# Now compare
same = df1_norm.equals(df2_norm)
print("Same?", same)

