from MySQLdb import _mysql

db=_mysql.connect(host="localhost",user="python",passwd = 'py' ,db="tpch")

print('done')

