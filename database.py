import sqlite3

def save_results(data):

    conn = sqlite3.connect("ssd_results.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        Temperature INTEGER,
        Wear_Level INTEGER,
        Read_Errors INTEGER,
        anomaly INTEGER
    )
    """)

    data.to_sql("results", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    print("Results saved to database")