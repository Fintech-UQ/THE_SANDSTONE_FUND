import pandas as pd




df=pd.read_csv('prices250.txt', sep='\s+', header=None, index_col=None)
# the T attribute swaps the rows and columns so the rows are now the stock prices
data = df.values.T
print(data)
print(data.T)



def convert_txt_file_to_csv(txt_file_name, new_csv_file_name):
    # e.g. file_name = 'prices250.txt'
    # e.g. new_csv_file_name = 'prices250.csv'
    # NOTE: the txt file must be in the same directory as this file
    read_file=pd.read_csv(txt_file_name,sep='\s+', header=None, index_col=None)
    read_file.to_csv(new_csv_file_name, header=None,index=None)
