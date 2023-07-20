from sqlalchemy import create_engine
import pandas as pd

def query_dataset(path):
    # Querying from database using sql query and convert it to dataframe ================
    # Note: Please edit the directory of the database base on your storage location
    path = 'sqlite:///' + path
    #print(path)
    engine = create_engine(path)
    print('Data Querying ============================')
    print('Accessed database with Table names:', engine.table_names())

    sql = "SELECT * FROM failure"
    dataframe = pd.read_sql(sql,con=engine)
    print('Dataset extracted from table.','\n')

    return dataframe