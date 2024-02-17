import base64
from io import BytesIO
import functions
import numpy as np
import streamlit as st
import pandas as pd
import functions as fc

def data_processing(df_name,df, n_rows,n_cols,sett_hints, user_precision,in_wid_change):
#++++++++++++++++++++++
    # DATA PROCESSING

    # Settings for data processing
    #-------------------------------------

    #st.subheader("Data processing")
    dev_expander_dm_sb = st.expander("Specify data processing preferences", expanded = False)
    with dev_expander_dm_sb:
        
        n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
        n_rows_wNAs_pre_processing = "No"
        if n_rows_wNAs > 0:
            n_rows_wNAs_pre_processing = "Yes"
            a1, a2, a3 = st.columns(3)
        else: a1, a3 = st.columns(2)
        
        sb_DM_dImp_num = None 
        sb_DM_dImp_other = None
        sb_DM_delRows=None
        sb_DM_keepRows=None
        
        with a1:
            #--------------------------------------------------------------------------------------
            # DATA CLEANING

            st.markdown("**Data cleaning**")

            # Delete rows
            delRows =st.selectbox('Delete rows with index ...', options=['-', 'greater', 'greater or equal', 'smaller', 'smaller or equal', 'equal', 'between'], on_change=in_wid_change)
            if delRows!='-':                                
                if delRows=='between':
                    row_1=st.number_input('Lower limit is', value=0, step=1, min_value= 0, max_value=len(df)-1, on_change=in_wid_change)
                    row_2=st.number_input('Upper limit is', value=2, step=1, min_value= 0, max_value=len(df)-1, on_change=in_wid_change)
                    if (row_1 + 1) < row_2 :
                        sb_DM_delRows=df.index[(df.index > row_1) & (df.index < row_2)]
                    elif (row_1 + 1) == row_2 : 
                        st.warning("WARNING: No row is deleted!")
                    elif row_1 == row_2 : 
                        st.warning("WARNING: No row is deleted!")
                    elif row_1 > row_2 :
                        st.error("ERROR: Lower limit must be smaller than upper limit!")  
                        return                   
                elif delRows=='equal':
                    sb_DM_delRows = st.multiselect("to...", df.index, on_change=in_wid_change)
                else:
                    row_1=st.number_input('than...', step=1, value=1, min_value = 0, max_value=len(df)-1, on_change=in_wid_change)                    
                    if delRows=='greater':
                        sb_DM_delRows=df.index[df.index > row_1]
                        if row_1 == len(df)-1:
                            st.warning("WARNING: No row is deleted!") 
                    elif delRows=='greater or equal':
                        sb_DM_delRows=df.index[df.index >= row_1]
                        if row_1 == 0:
                            st.error("ERROR: All rows are deleted!")
                            return
                    elif delRows=='smaller':
                        sb_DM_delRows=df.index[df.index < row_1]
                        if row_1 == 0:
                            st.warning("WARNING: No row is deleted!") 
                    elif delRows=='smaller or equal':
                        sb_DM_delRows=df.index[df.index <= row_1]
                        if row_1 == len(df)-1:
                            st.error("ERROR: All rows are deleted!")
                            return
                if sb_DM_delRows is not None:
                    df = df.loc[~df.index.isin(sb_DM_delRows)]
                    no_delRows=n_rows-df.shape[0]

            # Keep rows
            keepRows =st.selectbox('Keep rows with index ...', options=['-', 'greater', 'greater or equal', 'smaller', 'smaller or equal', 'equal', 'between'], on_change=in_wid_change)
            if keepRows!='-':                                
                if keepRows=='between':
                    row_1=st.number_input('Lower limit is', value=0, step=1, min_value= 0, max_value=len(df)-1, on_change=in_wid_change)
                    row_2=st.number_input('Upper limit is', value=2, step=1, min_value= 0, max_value=len(df)-1, on_change=in_wid_change)
                    if (row_1 + 1) < row_2 :
                        sb_DM_keepRows=df.index[(df.index > row_1) & (df.index < row_2)]
                    elif (row_1 + 1) == row_2 : 
                        st.error("ERROR: No row is kept!")
                        return
                    elif row_1 == row_2 : 
                        st.error("ERROR: No row is kept!")
                        return
                    elif row_1 > row_2 :
                        st.error("ERROR: Lower limit must be smaller than upper limit!")  
                        return                   
                elif keepRows=='equal':
                    sb_DM_keepRows = st.multiselect("to...", df.index, on_change=in_wid_change)
                else:
                    row_1=st.number_input('than...', step=1, value=1, min_value = 0, max_value=len(df)-1, on_change=in_wid_change)                    
                    if keepRows=='greater':
                        sb_DM_keepRows=df.index[df.index > row_1]
                        if row_1 == len(df)-1:
                            st.error("ERROR: No row is kept!") 
                            return
                    elif keepRows=='greater or equal':
                        sb_DM_keepRows=df.index[df.index >= row_1]
                        if row_1 == 0:
                            st.warning("WARNING: All rows are kept!")
                    elif keepRows=='smaller':
                        sb_DM_keepRows=df.index[df.index < row_1]
                        if row_1 == 0:
                            st.error("ERROR: No row is kept!") 
                            return
                    elif keepRows=='smaller or equal':
                        sb_DM_keepRows=df.index[df.index <= row_1]
                if sb_DM_keepRows is not None:
                    df = df.loc[df.index.isin(sb_DM_keepRows)]
                    no_keptRows=df.shape[0]

            # Delete columns
            sb_DM_delCols = st.multiselect("Select columns to delete ", df.columns, on_change=in_wid_change)
            df = df.loc[:,~df.columns.isin(sb_DM_delCols)]

            # Keep columns
            sb_DM_keepCols = st.multiselect("Select columns to keep", df.columns, on_change=in_wid_change)
            if len(sb_DM_keepCols) > 0:
                df = df.loc[:,df.columns.isin(sb_DM_keepCols)]

            # Delete duplicates if any exist
            if df[df.duplicated()].shape[0] > 0:
                sb_DM_delDup = st.selectbox("Delete duplicate rows ", ["No", "Yes"], on_change=in_wid_change)
                if sb_DM_delDup == "Yes":
                    n_rows_dup = df[df.duplicated()].shape[0]
                    df = df.drop_duplicates()
            elif df[df.duplicated()].shape[0] == 0:   
                sb_DM_delDup = "No"    
                
            # Delete rows with NA if any exist
            n_rows_wNAs = df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0]
            if n_rows_wNAs > 0:
                sb_DM_delRows_wNA = st.selectbox("Delete rows with NAs ", ["No", "Yes"], on_change=in_wid_change)
                if sb_DM_delRows_wNA == "Yes": 
                    df = df.dropna()
            elif n_rows_wNAs == 0: 
                sb_DM_delRows_wNA = "No"   

            # Filter data
            st.markdown("**Data filtering**")
            filter_var = st.selectbox('Filter your data by a variable...', list('-')+ list(df.columns), on_change=in_wid_change)
            if filter_var !='-':
                
                if df[filter_var].dtypes=="int64" or df[filter_var].dtypes=="float64": 
                    if df[filter_var].dtypes=="float64":
                        filter_format="%.8f"
                    else:
                        filter_format=None

                    user_filter=st.selectbox('Select values that are ...', options=['greater','greater or equal','smaller','smaller or equal', 'equal','between'], on_change=in_wid_change)
                                            
                    if user_filter=='between':
                        filter_1=st.number_input('Lower limit is', format=filter_format, value=df[filter_var].min(), min_value=df[filter_var].min(), max_value=df[filter_var].max(), on_change=in_wid_change)
                        filter_2=st.number_input('Upper limit is', format=filter_format, value=df[filter_var].max(), min_value=df[filter_var].min(), max_value=df[filter_var].max(), on_change=in_wid_change)
                        #reclassify values:
                        if filter_1 < filter_2 :
                            df = df[(df[filter_var] > filter_1) & (df[filter_var] < filter_2)] 
                            if len(df) == 0:
                                st.error("ERROR: No data available for the selected limits!")  
                                return        
                        elif filter_1 >= filter_2 :
                            st.error("ERROR: Lower limit must be smaller than upper limit!")  
                            return                    
                    elif user_filter=='equal':                            
                        filter_1=st.multiselect('to... ', options=df[filter_var].values, on_change=in_wid_change)
                        if len(filter_1)>0:
                            df = df.loc[df[filter_var].isin(filter_1)]

                    else:
                        filter_1=st.number_input('than... ',format=filter_format, value=df[filter_var].min(), min_value=df[filter_var].min(), max_value=df[filter_var].max(), on_change=in_wid_change)
                        #reclassify values:
                        if user_filter=='greater':
                            df = df[df[filter_var] > filter_1]
                        elif user_filter=='greater or equal':
                            df = df[df[filter_var] >= filter_1]        
                        elif user_filter=='smaller':
                            df= df[df[filter_var]< filter_1] 
                        elif user_filter=='smaller or equal':
                            df = df[df[filter_var] <= filter_1]
            
                        if len(df) == 0:
                            st.error("ERROR: No data available for the selected value!")
                            return 
                        elif len(df) == n_rows:
                            st.warning("WARNING: Data are not filtered for this value!")         
                else:                  
                    filter_1=st.multiselect('Filter your data by a value...', (df[filter_var]).unique(), on_change=in_wid_change)
                    if len(filter_1)>0:
                        df = df.loc[df[filter_var].isin(filter_1)]
            
        if n_rows_wNAs_pre_processing == "Yes":
            with a2: 
                #--------------------------------------------------------------------------------------
                # DATA IMPUTATION

                # Select data imputation method (only if rows with NA not deleted)
                if sb_DM_delRows_wNA == "No" and n_rows_wNAs > 0:
                    st.markdown("**Data imputation**")
                    sb_DM_dImp_choice = st.selectbox("Replace entries with NA ", ["No", "Yes"], on_change=in_wid_change)
                    if sb_DM_dImp_choice == "Yes":
                        # Numeric variables
                        sb_DM_dImp_num = st.selectbox("Imputation method for numeric variables ", ["Mean", "Median", "Random value"], on_change=in_wid_change)
                        # Other variables
                        sb_DM_dImp_other = st.selectbox("Imputation method for other variables ", ["Mode", "Random value"], on_change=in_wid_change)
                        df = fc.data_impute(df, sb_DM_dImp_num, sb_DM_dImp_other)
                else:
                    st.markdown("**Data imputation**")
                    st.write("")
                    st.info("No NAs in data set!")
        
        with a3:
            #--------------------------------------------------------------------------------------
            # DATA TRANSFORMATION

            st.markdown("**Data transformation**")
            
                                            
            # Select columns for different transformation types
            transform_options = df.select_dtypes([np.number]).columns
            numCat_options = df.columns
            ohe_options = df.select_dtypes(include=['object', 'category']).columns
                                                    
            sb_DM_dTrans_ohe = st.multiselect("Select columns for one hot encoding", ohe_options, on_change=in_wid_change)
            if len(sb_DM_dTrans_ohe) > 0:
                df = fc.var_transform_ohe(df, sb_DM_dTrans_ohe)  
            sb_DM_dTrans_log = st.multiselect("Select columns to transform with log ", transform_options, on_change=in_wid_change)
            if sb_DM_dTrans_log is not None: 
                df = fc.var_transform_log(df, sb_DM_dTrans_log)
            sb_DM_dTrans_sqrt = st.multiselect("Select columns to transform with sqrt ", transform_options, on_change=in_wid_change)
            if sb_DM_dTrans_sqrt is not None: 
                df = fc.var_transform_sqrt(df, sb_DM_dTrans_sqrt)
            sb_DM_dTrans_square = st.multiselect("Select columns for squaring ", transform_options, on_change=in_wid_change)
            if sb_DM_dTrans_square is not None: 
                df = fc.var_transform_square(df, sb_DM_dTrans_square)
            sb_DM_dTrans_cent = st.multiselect("Select columns for centering ", transform_options, on_change=in_wid_change)
            if sb_DM_dTrans_cent is not None: 
                df = fc.var_transform_cent(df, sb_DM_dTrans_cent)
            sb_DM_dTrans_stand = st.multiselect("Select columns for standardization ", transform_options, on_change=in_wid_change)
            if sb_DM_dTrans_stand is not None: 
                df = fc.var_transform_stand(df, sb_DM_dTrans_stand)
            sb_DM_dTrans_norm = st.multiselect("Select columns for normalization ", transform_options, on_change=in_wid_change)
            if sb_DM_dTrans_norm is not None: 
                df = fc.var_transform_norm(df, sb_DM_dTrans_norm)
            sb_DM_dTrans_numCat = st.multiselect("Select columns for numeric categorization ", numCat_options, on_change=in_wid_change)
            if sb_DM_dTrans_numCat:
                if not df[sb_DM_dTrans_numCat].columns[df[sb_DM_dTrans_numCat].isna().any()].tolist(): 
                    sb_DM_dTrans_numCat_sel = st.multiselect("Select variables for manual categorization ", sb_DM_dTrans_numCat, on_change=in_wid_change)
                    if sb_DM_dTrans_numCat_sel:
                        for var in sb_DM_dTrans_numCat_sel:
                            if df[var].unique().size > 5: 
                                st.error("ERROR: Selected variable has too many categories (>5): " + str(var))
                                return
                            else:
                                manual_cats = pd.DataFrame(index = range(0, df[var].unique().size), columns=["Value", "Cat"])
                                text = "Category for "
                                # Save manually selected categories
                                for i in range(0, df[var].unique().size):
                                    text1 = text + str(var) + ": " + str(sorted(df[var].unique())[i])
                                    man_cat = st.number_input(text1, value = 0, min_value=0, on_change=in_wid_change)
                                    manual_cats.loc[i]["Value"] = sorted(df[var].unique())[i]
                                    manual_cats.loc[i]["Cat"] = man_cat
                                
                                new_var_name = "numCat_" + var
                                new_var = pd.DataFrame(index = df.index, columns = [new_var_name])
                                for c in df[var].index:
                                    if pd.isnull(df[var][c]) == True:
                                        new_var.loc[c, new_var_name] = np.nan
                                    elif pd.isnull(df[var][c]) == False:
                                        new_var.loc[c, new_var_name] = int(manual_cats[manual_cats["Value"] == df[var][c]]["Cat"])
                                df[new_var_name] = new_var.astype('int64')
                            # Exclude columns with manual categorization from standard categorization
                            numCat_wo_manCat = [var for var in sb_DM_dTrans_numCat if var not in sb_DM_dTrans_numCat_sel]
                            df = fc.var_transform_numCat(df, numCat_wo_manCat)
                    else:
                        df = fc.var_transform_numCat(df, sb_DM_dTrans_numCat)
                else:
                    col_with_na = df[sb_DM_dTrans_numCat].columns[df[sb_DM_dTrans_numCat].isna().any()].tolist()
                    st.error("ERROR: Please select columns without NAs: " + ', '.join(map(str,col_with_na)))
                    return
            else:
                sb_DM_dTrans_numCat = None
            sb_DM_dTrans_mult = st.number_input("Number of variable multiplications ", value = 0, min_value=0, on_change=in_wid_change)
            if sb_DM_dTrans_mult != 0: 
                multiplication_pairs = pd.DataFrame(index = range(0, sb_DM_dTrans_mult), columns=["Var1", "Var2"])
                text = "Multiplication pair"
                for i in range(0, sb_DM_dTrans_mult):
                    text1 = text + " " + str(i+1)
                    text2 = text + " " + str(i+1) + " "
                    mult_var1 = st.selectbox(text1, transform_options, on_change=in_wid_change)
                    mult_var2 = st.selectbox(text2, transform_options, on_change=in_wid_change)
                    multiplication_pairs.loc[i]["Var1"] = mult_var1
                    multiplication_pairs.loc[i]["Var2"] = mult_var2
                    fc.var_transform_mult(df, mult_var1, mult_var2)
            sb_DM_dTrans_div = st.number_input("Number of variable divisions ", value = 0, min_value=0, on_change=in_wid_change)
            if sb_DM_dTrans_div != 0:
                division_pairs = pd.DataFrame(index = range(0, sb_DM_dTrans_div), columns=["Var1", "Var2"]) 
                text = "Division pair"
                for i in range(0, sb_DM_dTrans_div):
                    text1 = text + " " + str(i+1) + " (numerator)"
                    text2 = text + " " + str(i+1) + " (denominator)"
                    div_var1 = st.selectbox(text1, transform_options, on_change=in_wid_change)
                    div_var2 = st.selectbox(text2, transform_options, on_change=in_wid_change)
                    division_pairs.loc[i]["Var1"] = div_var1
                    division_pairs.loc[i]["Var2"] = div_var2
                    fc.var_transform_div(df, div_var1, div_var2)

            data_transform=st.checkbox("Transform data in Excel?", value=False)
            if data_transform==True:
                st.info("Press the button to open your data in Excel. Don't forget to save your result as a csv or a txt file!")
                # Download link
                output = BytesIO()
                excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                df.to_excel(excel_file, sheet_name="data",index=False)    
                excel_file.close()
                excel_file = output.getvalue()
                b64 = base64.b64encode(excel_file)
                dl_file_name = "Data_transformation__" + df_name + ".xlsx"
                st.markdown(
                    f"""
                <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Transform your data in Excel</a>
                """,
                unsafe_allow_html=True)
            st.write("")   
                
        #--------------------------------------------------------------------------------------
        # PROCESSING SUMMARY
        
        if st.checkbox('Show a summary of my data processing preferences ', value = False): 
            st.markdown("Summary of data changes:")

            #--------------------------------------------------------------------------------------
            # DATA CLEANING

            # Rows
            if sb_DM_delRows is not None and delRows!='-' :
                if no_delRows > 1:
                    st.write("-", no_delRows, " rows were deleted!")
                elif no_delRows == 1:
                    st.write("-",no_delRows, " row was deleted!")
                elif no_delRows == 0:
                    st.write("- No row was deleted!")
            else:
                st.write("- No row was deleted!")
            if sb_DM_keepRows is not None and keepRows!='-' :
                if no_keptRows > 1:
                    st.write("-", no_keptRows, " rows are kept!")
                elif no_keptRows == 1:
                    st.write("-",no_keptRows, " row is kept!")
                elif no_keptRows == 0:
                    st.write("- All rows are kept!")
            else:
                st.write("- All rows are kept!") 
            # Columns
            if len(sb_DM_delCols) > 1:
                st.write("-", len(sb_DM_delCols), " columns were deleted:", ', '.join(sb_DM_delCols))
            elif len(sb_DM_delCols) == 1:
                st.write("-",len(sb_DM_delCols), " column was deleted:", str(sb_DM_delCols[0]))
            elif len(sb_DM_delCols) == 0:
                st.write("- No column was deleted!")
            if len(sb_DM_keepCols) > 1:
                st.write("-", len(sb_DM_keepCols), " columns are kept:", ', '.join(sb_DM_keepCols))
            elif len(sb_DM_keepCols) == 1:
                st.write("-",len(sb_DM_keepCols), " column is kept:", str(sb_DM_keepCols[0]))
            elif len(sb_DM_keepCols) == 0:
                st.write("- All columns are kept!")
            # Duplicates
            if sb_DM_delDup == "Yes":
                if n_rows_dup > 1:
                    st.write("-", n_rows_dup, " duplicate rows were deleted!")
                elif n_rows_dup == 1:
                    st.write("-", n_rows_dup, "duplicate row was deleted!")
            else:
                st.write("- No duplicate row was deleted!")
            # NAs
            if sb_DM_delRows_wNA == "Yes":
                if n_rows_wNAs > 1:
                    st.write("-", n_rows_wNAs, "rows with NAs were deleted!")
                elif n_rows_wNAs == 1:
                    st.write("-", n_rows - n_rows_wNAs, "row with NAs was deleted!")
            else:
                st.write("- No row with NAs was deleted!")
            # Filter
            if filter_var != "-":
                if df[filter_var].dtypes=="int64" or df[filter_var].dtypes=="float64":
                    if isinstance(filter_1, list):
                        if len(filter_1) == 0:
                            st.write("-", " Data was not filtered!")
                        elif len(filter_1) > 0:
                            st.write("-", " Data filtered by:", str(filter_var))
                    elif filter_1 is not None:
                        st.write("-", " Data filtered by:", str(filter_var))
                    else:
                        st.write("-", " Data was not filtered!")
                elif len(filter_1)>0:
                    st.write("-", " Data filtered by:", str(filter_var))
                elif len(filter_1) == 0:
                    st.write("-", " Data was not filtered!")
            else:
                st.write("-", " Data was not filtered!")
                
            #--------------------------------------------------------------------------------------
            # DATA IMPUTATION

            if sb_DM_delRows_wNA == "No" and n_rows_wNAs > 0:
                st.write("- Data imputation method for numeric variables:", sb_DM_dImp_num)
                st.write("- Data imputation method for other variable types:", sb_DM_dImp_other)

            #--------------------------------------------------------------------------------------
            # DATA TRANSFORMATION

            #ohe
            if len(sb_DM_dTrans_ohe) > 1:
                st.write("-", len(sb_DM_dTrans_ohe), " columns were OHE-transformed:", ', '.join(sb_DM_dTrans_ohe))
            elif len(sb_DM_dTrans_ohe) == 1:
                st.write("-",len(sb_DM_dTrans_ohe), " column was OHE-transformed:", sb_DM_dTrans_ohe[0])
            elif len(sb_DM_dTrans_ohe) == 0:
                st.write("- No column was OHE-transformed!")
            # log
            if len(sb_DM_dTrans_log) > 1:
                st.write("-", len(sb_DM_dTrans_log), " columns were log-transformed:", ', '.join(sb_DM_dTrans_log))
            elif len(sb_DM_dTrans_log) == 1:
                st.write("-",len(sb_DM_dTrans_log), " column was log-transformed:", sb_DM_dTrans_log[0])
            elif len(sb_DM_dTrans_log) == 0:
                st.write("- No column was log-transformed!")
            # sqrt
            if len(sb_DM_dTrans_sqrt) > 1:
                st.write("-", len(sb_DM_dTrans_sqrt), " columns were sqrt-transformed:", ', '.join(sb_DM_dTrans_sqrt))
            elif len(sb_DM_dTrans_sqrt) == 1:
                st.write("-",len(sb_DM_dTrans_sqrt), " column was sqrt-transformed:", sb_DM_dTrans_sqrt[0])
            elif len(sb_DM_dTrans_sqrt) == 0:
                st.write("- No column was sqrt-transformed!")
            # square
            if len(sb_DM_dTrans_square) > 1:
                st.write("-", len(sb_DM_dTrans_square), " columns were squared:", ', '.join(sb_DM_dTrans_square))
            elif len(sb_DM_dTrans_square) == 1:
                st.write("-",len(sb_DM_dTrans_square), " column was squared:", sb_DM_dTrans_square[0])
            elif len(sb_DM_dTrans_square) == 0:
                st.write("- No column was squared!")
            # centering
            if len(sb_DM_dTrans_cent) > 1:
                st.write("-", len(sb_DM_dTrans_cent), " columns were centered:", ', '.join(sb_DM_dTrans_cent))
            elif len(sb_DM_dTrans_cent) == 1:
                st.write("-",len(sb_DM_dTrans_cent), " column was centered:", sb_DM_dTrans_cent[0])
            elif len(sb_DM_dTrans_cent) == 0:
                st.write("- No column was centered!")
            # standardize
            if len(sb_DM_dTrans_stand) > 1:
                st.write("-", len(sb_DM_dTrans_stand), " columns were standardized:", ', '.join(sb_DM_dTrans_stand))
            elif len(sb_DM_dTrans_stand) == 1:
                st.write("-",len(sb_DM_dTrans_stand), " column was standardized:", sb_DM_dTrans_stand[0])
            elif len(sb_DM_dTrans_stand) == 0:
                st.write("- No column was standardized!")
            # normalize
            if len(sb_DM_dTrans_norm) > 1:
                st.write("-", len(sb_DM_dTrans_norm), " columns were normalized:", ', '.join(sb_DM_dTrans_norm))
            elif len(sb_DM_dTrans_norm) == 1:
                st.write("-",len(sb_DM_dTrans_norm), " column was normalized:", sb_DM_dTrans_norm[0])
            elif len(sb_DM_dTrans_norm) == 0:
                st.write("- No column was normalized!")
            # numeric category
            if sb_DM_dTrans_numCat is not None:
                if len(sb_DM_dTrans_numCat) > 1:
                    st.write("-", len(sb_DM_dTrans_numCat), " columns were transformed to numeric categories:", ', '.join(sb_DM_dTrans_numCat))
                elif len(sb_DM_dTrans_numCat) == 1:
                    st.write("-",len(sb_DM_dTrans_numCat), " column was transformed to numeric categories:", sb_DM_dTrans_numCat[0])
            elif sb_DM_dTrans_numCat is None:
                st.write("- No column was transformed to numeric categories!")
            # multiplication
            if sb_DM_dTrans_mult != 0:
                st.write("-", "Number of variable multiplications: ", sb_DM_dTrans_mult)
            elif sb_DM_dTrans_mult == 0:
                st.write("- No variables were multiplied!")
            # division
            if sb_DM_dTrans_div != 0:
                st.write("-", "Number of variable divisions: ", sb_DM_dTrans_div)
            elif sb_DM_dTrans_div == 0:
                st.write("- No variables were divided!")
            st.write("")
            st.write("")   

    #------------------------------------------------------------------------------------------

    #++++++++++++++++++++++
    # UPDATED DATA SUMMARY   

    # Show only if changes were made
    if any(v for v in [sb_DM_delCols, sb_DM_dImp_num, sb_DM_dImp_other, sb_DM_dTrans_ohe, sb_DM_dTrans_log, sb_DM_dTrans_sqrt, sb_DM_dTrans_square, sb_DM_dTrans_cent, sb_DM_dTrans_stand, sb_DM_dTrans_norm, sb_DM_dTrans_numCat ] if v is not None) or sb_DM_delDup == "Yes" or sb_DM_delRows_wNA == "Yes" or sb_DM_dTrans_mult != 0 or sb_DM_dTrans_div != 0 or filter_var != "-" or delRows!='-' or keepRows!='-' or len(sb_DM_keepCols) > 0:
        dev_expander_dsPost = st.expander("Explore cleaned and transformed data info and stats ", expanded = False)
        with dev_expander_dsPost:
            if df.shape[1] > 0 and df.shape[0] > 0:

                # Show cleaned and transformed data & data info
                df_summary_post = fc.data_summary(df)
                if st.checkbox("Show cleaned and transformed data ", value = False):  
                    n_rows_post = df.shape[0]
                    n_cols_post = df.shape[1]
                    st.dataframe(df)
                    st.write("Data shape: ", n_rows_post, "rows and ", n_cols_post, "columns")
                
                    # Download transformed data:
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    df.to_excel(excel_file, sheet_name="Clean. and transf. data")
                    excel_file.close()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    dl_file_name = "CleanedTransfData__" + df_name + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download cleaned and transformed data</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")
                
                if df[df.duplicated()].shape[0] > 0 or df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                    check_nasAnddupl2 = st.checkbox("Show duplicates and NAs info (processed) ", value = False) 
                    if check_nasAnddupl2:
                        index_c = []
                        for c in df.columns:
                            for r in df.index:
                                if pd.isnull(df[c][r]):
                                    index_c.append(r)      
                        if df[df.duplicated()].shape[0] > 0:
                            st.write("Number of duplicates: ", df[df.duplicated()].shape[0])
                            st.write("Duplicate row index: ", ', '.join(map(str,list(df.index[df.duplicated()]))))
                        if df.iloc[list(pd.unique(np.where(df.isnull())[0]))].shape[0] > 0:
                            st.write("Number of rows with NAs: ", len(pd.unique(sorted(index_c))))
                            st.write("Rows with NAs: ", ', '.join(map(str,list(pd.unique(sorted(index_c))))))

                # Show cleaned and transformed variable info
                if st.checkbox("Show cleaned and transformed variable info ", value = False): 
                    st.write(df_summary_post["Variable types"])

                # Show summary statistics (cleaned and transformed data)
                if st.checkbox('Show summary statistics (cleaned and transformed data) ', value = False):
                    st.write(df_summary_post["ALL"].style.format(precision=user_precision))

                    # Download link
                    output = BytesIO()
                    excel_file = pd.ExcelWriter(output, engine="xlsxwriter")
                    df.to_excel(excel_file, sheet_name="cleaned_data")
                    df_summary_post["Variable types"].to_excel(excel_file, sheet_name="cleaned_variable_info")
                    df_summary_post["ALL"].to_excel(excel_file, sheet_name="cleaned_summary_statistics")
                    excel_file.close()
                    excel_file = output.getvalue()
                    b64 = base64.b64encode(excel_file)
                    dl_file_name = "Cleaned data summary statistics_multi_" + df_name + ".xlsx"
                    st.markdown(
                        f"""
                    <a href="data:file/excel_file;base64,{b64.decode()}" id="button_dl" download="{dl_file_name}">Download cleaned data summary statistics</a>
                    """,
                    unsafe_allow_html=True)
                    st.write("")

                    if fc.get_mode(df).loc["n_unique"].any():
                        st.caption("** Mode is not unique.")
                    if sett_hints:
                        st.info(str(fc.learning_hints("de_summary_statistics")))  
            else: 
                st.error("ERROR: No data available for preprocessing!") 
                return
     
    #------------------------------------------------------------------------------------------
        
    return(df,sb_DM_dTrans_ohe, sb_DM_dTrans_log, sb_DM_dTrans_sqrt, sb_DM_dTrans_square, sb_DM_dTrans_cent, sb_DM_dTrans_stand, sb_DM_dTrans_norm, sb_DM_dTrans_numCat, sb_DM_dTrans_mult, sb_DM_dTrans_div )