import os,glob
import pandas as pd

m=int(input("levels: "))

cwd =os.getcwd()
cwd=cwd+"/";

for i in range(m+1):
	path=cwd
	all_files=glob.glob(os.path.join(path,"chain0"+str(i)+"_ind_samples*"))
	new_files=glob.glob(os.path.join(path,"chain0"+str(i)+"_ind_samples_*"))
	df_merged=(pd.read_csv(f, sep=',') for f in all_files)
	df_merged=pd.concat(df_merged, ignore_index=True)
	df_merged.to_csv(f"chain0{i}_ind_samples",index=False)

	for f in new_files:
		os.remove(f)

for i in range(1,m+1):
	for j in range(m-i+1):
		path=cwd
		all_files=glob.glob(os.path.join(path,"chain"+str(i)+str(j)+"Lower_ind_samples*"))
		new_files=glob.glob(os.path.join(path,"chain"+str(i)+str(j)+"Lower_ind_samples_*"))
		df_merged=(pd.read_csv(f, sep=',') for f in all_files)
		df_merged=pd.concat(df_merged, ignore_index=True)
		df_merged.to_csv(f"chain{i}{j}Lower_ind_samples",index=False)

		for f in new_files:
			os.remove(f)

		path=cwd
		all_files=glob.glob(os.path.join(path,"chain"+str(i)+str(j)+"Upper_ind_samples*"))
		new_files=glob.glob(os.path.join(path,"chain"+str(i)+str(j)+"Upper_ind_samples_*"))
		df_merged=(pd.read_csv(f, sep=',') for f in all_files)
		df_merged=pd.concat(df_merged, ignore_index=True)
		df_merged.to_csv(f"chain{i}{j}Upper_ind_samples",index=False)

		for f in new_files:
			os.remove(f)