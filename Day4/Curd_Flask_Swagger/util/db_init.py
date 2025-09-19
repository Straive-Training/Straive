from util.common import get_db_connection, get_connection_no_db


def create_database_if_not_exists():
    conn = get_connection_no_db()
    cursor = conn.cursor()
    cursor.execute(
        "CREATE DATABASE IF NOT EXISTS bookdb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    cursor.close()
    conn.close()
    print("Database checked/created.")


def init_db():
    # Create DB first if needed
    create_database_if_not_exists()

    # Then create tables inside DB
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255) NOT NULL
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print("Tables checked/created.")


if __name__ == '__main__':
    init_db()
