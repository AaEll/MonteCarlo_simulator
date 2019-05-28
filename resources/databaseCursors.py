class dataFrameCursor():
    '''wrapper for dataframe to use same API as DictCursor for databases'''
    
    def __init__(self, dataframes, table_names ):
        self.tables = {}
        for df, name  in zip(dataframes,table_names):
            self.tables[name] = df
    
    def dictionary_iterator(self, table_name):
        for index, row in self.tables[table_name].iterrows():
            yield row
            
    def execute(self, sql, table):
        assert(sql=='''select * from ?''')
        self.next_table = table
        
    def fetchall(self): 
        return self.dictionary_iterator(self.next_table)

