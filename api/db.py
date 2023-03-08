import os
import mysql.connector.pooling

class ConnectionPool:
    def __init__(self, size=5):
        self.size = size
        self.pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="api_pool",
            pool_size=size,
            host=os.environ["MYSQL_HOST"],
            user=os.environ["MYSQL_USER"],
            passwd=os.environ["MYSQL_ROOT_PASSWORD"],
            db=os.environ["MYSQL_DATABASE"],
        )

    def get_connection(self):
        return self.pool.get_connection()

    def close_all_connections(self):
        self.pool.closeall()
