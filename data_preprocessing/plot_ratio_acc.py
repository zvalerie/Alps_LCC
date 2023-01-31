import pandas as pd
import matplotlib.pyplot as plt

idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df = pd.read_csv('/data/xiaolong/master_thesis/ratio_acc.csv')
print(df.head())
new_df = df[df['class'].isin([4])]
raw_df=new_df.groupby('ratio', as_index=False).mean()

# l1=plt.plot(raw_df['class'],raw_df['accuracy'],'r--',label='type1')
# plt.show()