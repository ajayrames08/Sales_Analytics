import pandas as pd
import numpy as np
import timeit

# This method is used to perform left join
# Input: Left Table, Right Table, Foreign key on left table, Foreign Key on right table
# Output: Left Joined Data

def left_join(left_data,right_data,common_identifier,Identifier):
    left_data=pd.merge(left_data,right_data,how='left',left_on=common_identifier,right_on=Identifier)
    left_data=left_data.reset_index(drop=True)
    return left_data
# This method is used to take unique items visited in  the same order
# Input: List of Items that are not unique
# Output: List of items preserved in same order and it is unique
def dedupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)
def main():
    start = timeit.default_timer()
    # Reading Data
    data=pd.read_csv("/Users/aj/PycharmProjects/Stepchange_Data_Preperation/StepChange_1.csv")
    stage_move=pd.read_csv("/Users/aj/PycharmProjects/Stepchange_Data_Preperation/StageMove.csv")
    stage=pd.read_csv("/Users/aj/PycharmProjects/Stepchange_Data_Preperation/Stage.csv")
    lead=pd.read_csv("/Users/aj/PycharmProjects/Stepchange_Data_Preperation/Lead.csv")
    lead_source=pd.read_csv("/Users/aj/PycharmProjects/Stepchange_Data_Preperation/LeadSource.csv")
    contact=pd.read_csv("/Users/aj/PycharmProjects/Stepchange_Data_Preperation/Contact.csv")
    contact_action=pd.read_csv("/Users/aj/PycharmProjects/Stepchange_Data_Preperation/ContactAction.csv")
    #Extracting Priority as one of the features
    contact_id_list=['Id','Priority' ]
    contact_action=contact_action[contact_id_list]
    new_data_joined=left_join(data,contact_action,'Opportunity Id','Id')
    new_data_joined = new_data_joined[np.isfinite(new_data_joined['Id'])]
    new_data_joined['Id']=new_data_joined['Id'].astype(int)
    opportunity_id=new_data_joined['Id'].tolist()
    #Extracting Unique Stage paths's as features
    columns = ['Id','StagePath']
    stage_path=pd.DataFrame(columns=columns)
    for i in opportunity_id:
        text=''
        for index,j in stage_move.iterrows():
            if (i==j['OpportunityId']):
                text+=str(','+str(j['MoveFromStage'])+','+str(j['MoveToStage']))
        stage_path=stage_path.append({'Id': i,'StagePath':text},ignore_index=True)
    new_data_joined=left_join(new_data_joined,stage_path,'Id','Id')
    list_stage=[]
    for i in opportunity_id:
        for index,j in new_data_joined.iterrows():
            if(i==j['Id']):
                stages=j['StagePath'].split(",")
                stage_list=list(dedupe(stages))
                stage_list.pop(0)
                list_stage.append((stage_list[:]))
    len_list=[]
    # Extracting No. of stages visited
    for sub_list in list_stage:
        length=len(sub_list)
        len_list.append(length)
    new_data_joined['UniqueStageList']=list_stage
    new_data_joined['Length']=len_list
    # Calcuating age and amount
    new_data_joined['Created Date']=pd.to_datetime(new_data_joined['Created Date'],format='%d/%m/%y')
    new_data_joined['Close Date']=pd.to_datetime(new_data_joined['Close Date'],format='%d/%m/%y')
    new_data_joined['Age']=(new_data_joined['Close Date']-new_data_joined['Created Date'])/np.timedelta64(1, 'D')
    new_data_joined['Amount']=new_data_joined['Amount High']+new_data_joined['Amount Low']
    del new_data_joined['Id']
    del new_data_joined['StagePath']
    # Saving the data frame to a file called StepChangeData_1.csv
    new_data_joined.to_csv("StepChangeData_1.csv",index=False)
    stop = timeit.default_timer()
    print "Algorithm Running Time for pre-processing:"
    print stop-start
if __name__ == "__main__": main()














