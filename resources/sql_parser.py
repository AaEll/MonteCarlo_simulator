from pyparsing import Word, delimitedList, Optional, \
    Group, alphas, alphanums, Forward, oneOf, quotedString, \
    infixNotation, opAssoc, ParseResults, \
    ZeroOrMore, restOfLine, CaselessKeyword, pyparsing_common as ppc
import matplotlib.pyplot as plt
import copy
import networkx as nx

class sqlParser():
    # define SQL tokens
    def create_sql(self):
        selectStmt = Forward()
        SELECT, FROM, WHERE, AND, OR, IN, IS, NOT, NULL, COUNT, AVG, MIN, MAX, SUM, AS = map(CaselessKeyword, 
            "select from where and or in is not null count avg min max sum as".split())
        NOT_NULL = NOT + NULL

        ident          = Word( alphas, alphanums + "_$" ).setName("identifier")
        alias          = delimitedList(ident, ".", combine=True).setName("alias")
        alias.addParseAction(ppc.upcaseTokens)
        columnName     = delimitedList(ident, ".", combine=True).setName("column name")
        columnName.addParseAction(ppc.upcaseTokens)
        columnNameList = Group( delimitedList(columnName))
        tableName      = delimitedList(ident, ".", combine=True).setName("table name")
        tableName.addParseAction(ppc.upcaseTokens)
        tableNameRalias= Group(
            (tableName("table") + AS + alias("alias")) |
            (tableName("table"))
            )

        tableNameList  = Group(delimitedList(tableNameRalias))

        binop = oneOf("= != < > >= <= eq ne lt le gt ge", caseless=True)
        realNum = ppc.real()
        intNum = ppc.signed_integer()

        columnRval = realNum | intNum | quotedString | columnName # need to add support for alg expressions
        val = realNum | intNum | quotedString
        columnRstar = '*' | columnName # need to add support for alg expressions
        EquiJoin =     (columnName('col1') + '=' + columnName ('col2'))
        equalityPredicate = columnName('col1') + '=' + columnRval ('val')
        Predicates =     Group(( columnName('col1') + binop + columnRval ) |
            ( columnName('col1') + IN + Group("(" + delimitedList( columnRval ) + ")" )) |
            ( columnName('col1') + IN + Group("(" + selectStmt + ")" )) |
            ( columnName + IS + (NULL | NOT_NULL)))

        whereCondition = Group(
            EquiJoin ('equijoin') |
            equalityPredicate ('equalitypredicate') |
            Predicates ('otherPredicates')
            )

        whereCondition_sketch = Group(
            EquiJoin ('equijoin') |
            equalityPredicate ('equalitypredicate') 
            )

        Aggregates = Group(
             ((COUNT|AVG|MIN|MAX|SUM)("operator") + (Group ("("+columnName+")"))("operand") ) |
             (COUNT("operator") + Group("("+"*"+")")("operand"))
            )

        AggregateExpression = delimitedList(Aggregates)

        whereExpression_predicates = infixNotation(whereCondition_sketch,
            [
                (AND, 2, opAssoc.LEFT),
            ])

        whereExpression = infixNotation(whereCondition,
            [
                (NOT, 1, opAssoc.RIGHT),
                (AND, 2, opAssoc.LEFT),
                (OR, 2, opAssoc.LEFT),
            ])

        # define the grammar
        selectStmt <<= (SELECT + ((AggregateExpression)("aggregates") | ('*' | columnNameList)("columns")) +
                        FROM + tableNameList( "tables" ) +
                        WHERE +(whereExpression_predicates)("sketch_predicates")+ Optional(AND + (whereExpression)("ignored_predicates")))

        simpleSQL = selectStmt

        # define Oracle comment format, and ignore them
        oracleSqlComment = "--" + restOfLine
        simpleSQL.ignore( oracleSqlComment )
        return simpleSQL
    
    def __init__ (self):
        print("bug : parser cannot handle a singular predicate")
        self.simpleSQL = self.create_sql()
        
    def parseString(self,sqlStatement, **kwargs):
        return self.simpleSQL.parseString(sqlStatement,**kwargs)
    

def CreateJoinGraph(queryString):
    parser = sqlParser()
    ParseResults = parser.parseString(queryString,parseAll=True)
    if "columns" in ParseResults:
        assert(0) # cannot return result using AQP
        
    aggregate_list = []
    table_dict = {}
    equi_joins_set = set([])
    predicates_set = set([])
    
    for aggregate in ParseResults['aggregates']:
        aggregate_list.append(aggregate)
        
    for table in ParseResults['tables']:
        if 'alias' in table:
            name = table['alias']
        else:
            name = table['table']
        if name in table_dict:
            assert(0) # cannot have duplicate names for tables
        else:
            table_dict[name] = table['table']
            
    for predicate in ParseResults['sketch_predicates']:
        print(predicate)
        print(type(predicate))

        if predicate == 'and':
            pass
        elif type(predicate) == type(ParseResults):
            if 'equijoin' in predicate:
                alias,column = predicate['col1'].split('.') # throws error if malformed column
                alias2,column2 = predicate['col2'].split('.')
                assert((alias in table_dict) and
                       (alias2 in table_dict) and
                       (alias != alias2))
                equi_joins_set.add( (alias,column, alias2,column2) )
            elif 'equalitypredicate' in predicate:
                alias,column = predicate['col1'].split('.')
                assert(alias in table_dict)
                value = predicate['val']
                predicates_set.add( (alias,column,value) )
            else:
                assert(0) # unexpected value in sketch_predicates
        else:
            print(predicate)
            print(type(predicate))
            assert(0) # unexpected value in sketch_predicates
    
    G = nx.MultiGraph()
    for ident, table_name in table_dict.items():
        G.add_node(ident, table_name = table_name)
    for equi_join in equi_joins_set:
        alias, col, alias2, col2 = equi_join
        G.add_edge(alias,alias2,equijoin = {alias:col, alias2:col2})
        #G[alias][alias2]['equijoin'] = {alias:col, alias2:col2}
    if predicates_set:
        G.add_node('__imaginary_table__')
        for predicate in predicates_set:
            alias, col, val = predicate
            G.add_edge('__imaginary_table__',alias,predicate = {alias : col, '__val__' : str(val).strip("\"\'")})
            #G['__imaginary_table__'][alias]['predicate'] = {alias : col, '__val__' : str(val)}
    return G

def drawGraph ( G):
    fig, ax = plt.subplots()
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])

    pos = nx.spring_layout(G)
    pos2 = copy.deepcopy(pos) 
    for p in pos2:  # raise text positions
        pos2[p][1] += 0.5


    nx.draw_networkx_edge_labels(G,pos = pos2,ax = ax)
    nx.draw(G, pos = pos,ax = ax,width = 3, font_size=16, with_labels=True)
    plt.show()


if __name__ == '__main__':
    G = CreateJoinGraph("select COUNT(*), avg (T2.mm), max(T1.ss) FROM table1 as T12,table2 where T12.name = table2.name AND T12.id = table2.id AND T12.wage = 4000")

    drawGraph(G)

    for node in G:
        print(node)
        for edge,e_dict in G[node].items():
            print(edge)        
            for e_id, e_attr in e_dict.items():
                print(e_attr)





