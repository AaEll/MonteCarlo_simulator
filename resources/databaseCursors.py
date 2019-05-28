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

from csv import reader
class tblCursor():
    '''wrapper for reader tpc-h .tbl with same API as DictCursor for databases'''
    
    def __init__(self, table_file_header ):
        
        self.next_table = ''
        self.headers = {}
        self.file_names = {}
        for table_name,file_name,header in table_file_header:
            self.headers[table_name] = header
            self.file_names[table_name] = file_name
    
    def dictionary_iterator(self, table_name):
        assert(table_name != '') # no such table
        
        with open(self.file_names[table_name], mode='r') as f:
            tbl_reader = reader(f,delimiter = '|')
            header = self.headers[table_name]
            for line in tbl_reader:
                yield {lm : r for lm,r in zip(header,line) }
            
    def execute(self, sql, table_name):
        assert(sql=='''select * from ?''')
        self.next_table = table_name
        
    def fetchall(self): 
        return self.dictionary_iterator(self.next_table)

