import sqlite3

class FileRegistry:
    def __init__(self, db_path="storage/file_registry.db"):
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False
        )

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                filename TEXT PRIMARY KEY
            )
        """)
        self.conn.commit()

    def is_processed(self, filename):
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM files WHERE filename=?", (filename,))
        return cur.fetchone() is not None

    def mark_processed(self, filename):
        self.conn.execute(
            "INSERT OR IGNORE INTO files VALUES (?)",
            (filename,)
        )
        self.conn.commit()

    def count_files(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM files")
        return cur.fetchone()[0]
