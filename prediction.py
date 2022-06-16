"""loading the validation file"""
validate_df

"""DRoping the unecessary cells in X"""
validate_df_X =validate_df.drop(["ID", "galaxy", "Predicted Well-Being Index"], axis=1)
# validate_df_y = validate_df["Predicted Well-Being Index"] ## validation files don't have y, since 
                                                           ## that is what we're looking for
validate_df_X = validate_df_X.iloc[:, valid_columns]
validate_df_X

#fill the empty cells with zero
validate_df_X = validate_df_X.apply(lambda x: x.fillna(0), axis = 0)
validate_df_X ## ok

X_data_pred = validate_df_X.iloc[:, 0:].values

##If the shape has 65 columns then we are good to go 
X_data_pred.shape ## 65 -> good

#Using Xgboost to predict the model
y_validate_pred_xg = model_xg.predict(X_data_pred)

y_validate_pred_xg

## changing the output to a dataframe
y_valid = pd.DataFrame(y_validate_pred_xg)
y_valid

## droping the predicted well being index column so that I can add the new predicted ones
validate_df_n=validate_df.drop(["Predicted Well-Being Index"], axis=1)
validate_df_n["Predicted Well-Being Index"]=y_valid
validate_df_n


#selecting specific rows in the dataframe
validate_df_print= validate_df_n.loc[:,["ID" ,"Predicted Well-Being Index"]]
validate_df_print
