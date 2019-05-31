import hashlib
import re, string 
from functools import reduce
from operator import mul


pattern = re.compile('^[a-zA-Z][a-zA-Z0-9]*$')


def adjListIterator(G,skipped_set=set([])):
    for node,n_attr in G.node.items():
        if node in skipped_set:
            continue
        yield (node,n_attr,G[node])
def edgeIterator(adjList):
    for edge,e_dict in adjList.items():
        # iterate again in the case of multi-edges
        for edge_number, e_attr in e_dict.items():
            yield (edge, e_attr)
def create_hash(value,seed):
    m = hashlib.md5(seed.encode('utf-8'))
    m.update(str(value).encode('utf-8'))
    return (int(m.hexdigest(),16)%2)*2 - 1

def partial_seed (e_attr,edge_type):
    return   '-'.join(e_attr[edge_type].keys())+'-'+'-'.join(e_attr[edge_type].values())

def create_full_seeds(partial_seed,num_replicas):
    ''' takes in edge-seed and appends it to unique values, (0,1,2, ..., k), for each replicas'''
    return  [str(x) + partial_seed for x in range(num_replicas)]

def average_signatures(signature_dict,num_replicas):
    ''' takes in signatures dict and returns estimate of query size'''
    return sum([reduce(mul, signature_tuple,1 ) for signature_tuple in zip(*signature_dict.values())])/num_replicas



def create_hashs(G,num_replicas,index_by = 'column'):
    ''' for a join graph G : create signatures for nodes in G for k independent hash functions, k = num_replicas.
        returns : dictionary of hash functions indexed by either index_by = column or index_by = seed
    '''
    signature_funcs = {}
    skipped_set = set(['__imaginary_table__'])
    
    for node,n_attr,adj_list in adjListIterator(G,skipped_set=skipped_set):
        table_name = n_attr['table_name']
        assert (pattern.match(table_name)) # check if valid sql table name        
        table_signature_funcs = {}
        for edge,e_attr in edgeIterator(adj_list):
            if 'equijoin' in e_attr:
                edge_type = 'equijoin'
            elif 'predicate' in e_attr:
                edge_type = 'predicate'
            else:
                assert(0) # malformed e_attr
            column_name = e_attr[edge_type][node]
            seed_str    = partial_seed (e_attr,edge_type)
            full_seeds  = create_full_seeds(seed_str,num_replicas)
            hashFuncs   = [lambda v,seed = seed: create_hash(v,seed) for seed in full_seeds]
            if index_by == 'column':
                if column_name in table_signature_funcs:
                    table_signature_funcs[column_name].append(hashFuncs)
                else:
                    table_signature_funcs[column_name]= [hashFuncs]
            elif index_by == 'seed':      
                if seed_str in table_signature_funcs:
                    assert(0) # should never be duplicate seeds
                table_signature_funcs[seed_str] = (column_name,hashFuncs)
            else:
                assert(0) # no other indexing is supported
                
        if table_name in signature_funcs:
            signature_funcs[table_name][node] = table_signature_funcs
        else:
            signature_funcs[table_name] = {node : table_signature_funcs}
    return signature_funcs



                    
def signature__imaginary_table__(adj_list, num_replicas):
    '''check for non-duplicate predicates and create signature of imaginary table'''
    signatures = [1]*num_replicas
    
    seen_alias_col = set([''])
    for edge, e_attr in edgeIterator(adj_list):
        seed_str    = partial_seed (e_attr ,'predicate')
        full_seeds  = create_full_seeds(seed_str,num_replicas)
        
        alias_col = ''
        for alias in e_attr['predicate']:
            if alias  != '__val__':
                alias_col = alias+e_attr['predicate'][alias]
                
        if alias_col in seen_alias_col:
            assert(alias_col!='') # malformed edge attributes
            assert(0) # cannot have multiple equality predicate on same relation col         
        else:
            seen_alias_col.add(alias_col)
        signatures = [sig*create_hash(e_attr['predicate']['__val__'],seed) for sig,seed in zip(signatures,full_seeds)]
    return signatures


def create_signatures(G,DictCursor,num_replicas):
    signatures_dict = {}
    initial_signatures = [1]*num_replicas
    
    signatures_dict['__imaginary_table__'] = signature__imaginary_table__(G['__imaginary_table__'],num_replicas)
    signature_funcs = create_hashs(G,num_replicas)
        
    for table in signature_funcs:
        sql='''select * from ?'''
        DictCursor.execute(sql, table)
        rows = DictCursor.fetchall()
        for alias in signature_funcs[table]:
            signatures_dict[alias] = [0]*num_replicas
        for dict_row in rows:
            for alias in signature_funcs[table]:
                signatures = initial_signatures
                for column in signature_funcs[table][alias]:
                    for hashfunctions in signature_funcs[table][alias][column]:
                        signatures = [sig*hashfunc(dict_row[column]) for sig,hashfunc in zip(signatures,hashfunctions) ]
                signatures_dict[alias]=[sig+delta for sig,delta in zip(signatures_dict[alias],signatures)]
    return signatures_dict

from itertools import combinations, chain

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def create_power_set_signatures(G,DictCursor,num_duplicate,num_replicas):
    signatures_dict = {}
    initial_signatures = [1]*num_replicas
    
    signature_funcs = create_hashs(G,num_replicas,index_by = 'seed')

    
    for table in signature_funcs:
        sql='''select * from ?'''
        DictCursor.execute(sql, table)
        rows = DictCursor.fetchall()
        powersets = {}
        for alias in signature_funcs[table]:
            signatures_dict[alias] = {}
            powersets[alias] = list(powerset(signature_funcs[table][alias].keys()))
            for edge_tuple in powersets[alias]:
                signatures_dict[alias][frozenset(edge_tuple)] = [0]*num_replicas
        for dict_row in rows:
            for alias in signature_funcs[table]:
                for edge_tuple in powersets[alias]:
                    signatures = initial_signatures
                    for edge_seed in edge_tuple:
                        column, hashfunctions = signature_funcs[table][alias][edge_seed]
                        signatures = [sig*hashfunc(dict_row[column]) for sig,hashfunc in zip(signatures,hashfunctions) ]
                    signatures_dict[alias][frozenset(edge_tuple)]=[sig+delta for sig,delta in zip(signatures_dict[alias][frozenset(edge_tuple)],signatures)]
    return (signatures_dict,num_replicas)

def create_seeds(G):
    seeds_dict = {}
    
    for node,n_attr in G.node.items():
        if node == '__imaginary_table__':
            pass # we do not construct dummy_table's signatures until runtime
        else:
            seeds_dict[node] = set([])
            for edge,e_dict in G[node].items():
                for e_id, e_attr in e_dict.items():
                    if 'equijoin' in e_attr:
                        column_name = e_attr['equijoin'][node]                    
                        seed_str = '-'.join(e_attr['equijoin'].keys())+'-'+'-'.join(e_attr['equijoin'].values())
                    elif 'predicate' in e_attr:
                        column_name = e_attr['predicate'][node]                    
                        seed_str = '-'.join(e_attr['predicate'].keys())+'-'+'-'.join(e_attr['predicate'].values())
                    else:
                        assert(0) # malformed attributes
                    if seed_str in seeds_dict[node]:
                        assert(0) # should never be duplicate seeds
                    seeds_dict[node].add(seed_str)
        
                    
    return seeds_dict

def lookup_signatures(G,powerset_signatures):
    
    # TODO rename duplicate tables
    signatures_dict, num_replicas = powerset_signatures
    
    imaginary_table_signature = signature__imaginary_table__(G['__imaginary_table__'],num_replicas)
    seeds_dict = create_seeds(G)
    final_signatures = {'__imaginary_table__' : imaginary_table_signature}
    
    for node,n_attr in G.node.items():
        if node == '__imaginary_table__':
            pass
        else:
            final_signatures[node] = signatures_dict[node][frozenset(seeds_dict[node])]
    return final_signatures
    


        
                    





if __name__ == '__main__':
    from sql_parser import CreateJoinGraph
    from databaseCursors import dataFrameCursor
    from pandas import DataFrame
    
    sql = """
        SELECT COUNT(*)
        FROM table1 as T11,
             table1 as T12,
             table2,
             table3
        WHERE T12.id = table2.id AND T12.name = T11.name AND T11.id = table3.id
                          AND T12.wage = 32000 """

    names =['TABLE1', 'TABLE2', 'TABLE3']
    df1 = DataFrame([[1,'jill',32000],
                       [2,'mariana',100000],
                       [3,'erica',32000],
                       [4,'sneha',32000],
                       [5,'hansle',32000]],columns = ['ID','NAME','WAGE'])


    df2 = DataFrame([[1,'jill','boxing'],
                       [2,'mariana','bad movies'],
                       [2,'mariana','pasta'],                    
                       [3,'erica','boxing'],
                       [4,'sneha','body building'],
                       [5,'hansle','swimming']], columns = ['ID','NAME','INTERESTS'])

    df3 = DataFrame([[1,'jill',25],
                       [2,'mariana',41],
                       [3,'erica',24],
                       [4,'sneha',31],
                       [5,'hansle',25]], columns = ['ID','NAME','AGE'])


    G = CreateJoinGraph(sql)

    Cursor = dataFrameCursor([df1,df2,df3],names)

    num_replicas = 5000
    queryresult = average_signatures(create_signatures(G,Cursor,num_replicas), num_replicas)
    
    print(sql)
    print('result : {}'.format(queryresult))

