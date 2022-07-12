
#%%%%Final solution to be submitted in Kaggle

# For each cv outer split, we obtain the competition test predictions from cross-validated ensemble model.
# In essence, this means that we have 10 (10 are cv outer splits) lists including the predictions 
# from the competition test set
# For this reason, we must obtain the most frequent prediction for each competition test set row/observation
# in order to take into advantage the cross-validated scores and hence, submit single predictions per observation

# Read
final_titanic_test_data=pd.read_csv(directory+"/test.csv",header=0, encoding="utf-8")

# calculate averages per list in order to obtain the probabilistic cross-validated out-of-sample predictions
y_out_sample_test_results_ensembled
y_out_sample_test_results_ensembled_averages = [sum(sub_list) / len(sub_list) for sub_list in zip(*y_out_sample_test_results_ensembled)]
y_out_sample_test_results_ensembled_averages_array =np.array(y_out_sample_test_results_ensembled_averages)
y_out_sample_test_results_ensembled_averages_array=np.where(y_out_sample_test_results_ensembled_averages_array > 0.5, 1,0)    

# Prepare the final dataframe for Kaggle Submission
y_out_sample_test_results_ensembled_averages_df=pd.DataFrame(y_out_sample_test_results_ensembled_averages_array).set_index(final_titanic_test_data["PassengerId"])
y_out_sample_test_results_ensembled_averages_df["PassengerId"]=y_out_sample_test_results_ensembled_averages_df.index 

y_out_sample_test_results_ensembled_averages_df.columns = ["Survived","PassengerId"]
y_out_sample_test_results_ensembled_averages_df=y_out_sample_test_results_ensembled_averages_df.reset_index(drop=True)
y_out_sample_test_results_ensembled_averages_df.to_csv('ensembled_predictions_xgb_rf.csv')#better predictions to belong in the top 55%


