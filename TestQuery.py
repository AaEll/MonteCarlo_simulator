import importlib
resources = importlib.import_module("resources")
from resources.sql_parser import CreateJoinGraph,drawGraph
from resources.databaseCursors import dataFrameCursor, tblCursor
from resources.signatures import create_signatures,create_hashs,\
                create_power_set_signatures, lookup_signatures,average_signatures


Part = ['PARTKEEY','NAME','MFGR','BRAND','TYPE',
        'SIZE','CONTAINER','RETAILPRICE','COMMENT']

LineItem = ['ORDERKEY','PARTKEY','SUPPKEY',
            'LINENUMBER','QUANTITY','EXTENDEDPRICE',
            'DISCOUNT','TAX','RETURNFLAG','LINESTATUS',
            'SHIPDATE','COMMITDATE','RECEIPTDATE',
            'SHIPINSTRUCT','SHIPMODE','COMMENT']
Region = ['REGIONKEY','NAME','COMMENT']
Orders = ['ORDERKEY','CUSTKEY','ORDERSTATUS',
          'TOTALPRICE','ORDERDATE','ORDERPRIORITY',
          'CLERK','SHIPPRIORITY','COMMENT']
PartSupp = ['PARTKEY','SUPPKEY','AVAILQTY',
            'SUPPLYCOST','COMMENT']
Customer = ['CUSTKEY','NAME','ADDRESS','PHONE',
            'ACCTBAL','MKTSEGMENT','COMMENT']
Nation = ['NATIONKEY','NAME','REGIONKEY','COMMENT']
Supplier = ['SUPPKEY','NAME','ADDRESS','NATIONKEY',
           'PHONE', 'ACCTBAL','COMMENT']


sql = """
    SELECT COUNT(*)
    FROM lineitem
    WHERE lineitem.QUANTITY = 11 AND lineitem.linenumber =14 """


G = CreateJoinGraph(sql)

table_file_header = [('PARTSUPP','data/partsupp.tbl',PartSupp),
                     ('LINEITEM','data/lineitem.tbl',LineItem),
                     ('REGION','data/region.tbl',Region),
                     ('ORDERS','data/orders.tbl',Orders),
                     ('PART','data/part.tbl',Part),
                     ('CUSTOMER','data/customer.tbl',Customer),
                     ('NATION','data/nation.tbl',Nation),
                     ('SUPPLIER','data/supplier.tbl',Supplier)]

Cursor = tblCursor(table_file_header)

_ = 2
with open('results/bias_replicas.txt','w') as f:
    f.write(sql+'\n\n')
    f.write('number of replicas  :  query result\n\n')
    for i in range(2,5):
        num_replicas = 2**(i*5)
        result = average_signatures(\
                    lookup_signatures(\
                       G, create_power_set_signatures(G,Cursor,_,num_replicas))\
                    ,num_replicas)
        f.write('{}  :  {}\n'.format(num_replicas, result))

print(sql)
print('\n')

sql_0 = """SELECT COUNT(*)
FROM customer, orders, lineitem
WHERE customer.custkey = orders.custkey
      AND lineitem.orderkey = orders.orderkey"""

G0 = CreateJoinGraph(sql_0)
print(average_signatures(lookup_signatures(G0,create_power_set_signatures(G,Cursor,_,num_replicas)),num_replicas))
print(sql_0)



