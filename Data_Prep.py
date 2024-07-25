from sqlalchemy import create_engine
from sqlalchemy import URL
from sqlalchemy import text
import pandas as pd
import json 
import csv
#hostname: 127.0.0.1
#port = 3306
#username: root
#password: root
engine = create_engine("mysql+pymysql://root:root@localhost:3306/MIMIC")


#MIMIC Data admissions


# with engine.connect() as conn:
#     query = text ("create table ehrs_diag as SELECT MIMIC.ehrs.ROW_ID, MIMIC.ehrs.SUBJECT_ID, MIMIC.ehrs.DESCRIPTION, MIMIC.ehrs.TEXT from MIMIC.ehrs   INNER JOIN MIMIC.diagnoses_icd on MIMIC.diagnoses_icd.SUBJECT_ID = MIMIC.ehrs.SUBJECT_ID")
#     result = conn.execute(query)
#     print("table created")
#     count = result.fetchone()[0]
#     print("results", count)

#select patinets with dementia


#read all the chunks and add to database 

# path ="../Data_chunks/"
# for i in range(99,100):
#     chunk = pd.read_csv(path+"chunk"+str(i)+".csv")
#     chunk = chunk.drop(chunk.columns[0], axis=1)
#     chunk.to_sql('ehrs', con=engine, if_exists='append', index=False)
#     print("chunk",i," added to database")



data = pd.read_csv("../ehr_subject_icd9/ehricdjs.csv")
print(data.head())