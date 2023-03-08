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
            passwd=os.environ["MYSQL_PASSWORD"],
            db=os.environ["MYSQL_DATABASE"],
        )

    def get_connection(self):
        return self.pool.get_connection()

    def close_all_connections(self):
        self.pool.closeall()


def setup_db(conn_pool):
    # Insert the serialized classifier into the database
    conn = conn_pool.get_connection()
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS db.classifiers (
                id INT NOT NULL AUTO_INCREMENT,
                model VARCHAR(255) NOT NULL,
                params TEXT,
                d INT NOT NULL,
                n_classes INT NOT NULL,
                n_trained INT NOT NULL DEFAULT 0,
                clf_bytes BLOB,
                PRIMARY KEY (id)
            )
            """
    )
    conn.commit()
    cur.execute("TRUNCATE db.classifiers")
    conn.commit()
    cur.close()
    conn.close()
