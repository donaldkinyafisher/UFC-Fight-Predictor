import pandas as pd
from sklearn.model_selection import train_test_split

#Load Dataset
ufc_dataset_df = pd.read_csv('data/ufc-master.csv')

#Select subset of data
#Drop date, location , country, fighter_names for the time being as they hold string data.
#Finish Details and total_fight_time_secs not available in dataset for an upcoming event

my_cols = ['R_odds', 'B_odds', 'R_ev', 'B_ev',
       'r_dec_odds', 'b_dec_odds', 'r_sub_odds',
       'b_sub_odds', 'r_ko_odds', 'b_ko_odds', 'Winner']
clean_ufc_df = pd.DataFrame(data= ufc_dataset_df, columns=my_cols)

#For 1st pass, just drop rows with nan values
clean_ufc_df = clean_ufc_df.dropna(axis = 'index')

#Split training and testing data, replacing  value strings with integers
X = clean_ufc_df.loc[:, clean_ufc_df.columns != 'Winner']
y = clean_ufc_df['Winner']
y = y.replace(to_replace='Blue', value=0)
y = y.replace(to_replace='Red', value=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)