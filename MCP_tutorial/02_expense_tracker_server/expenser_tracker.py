from fastmcp import FastMCP
import os, sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "expense_tracker.db")
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "database", "categories.json")
# print(DB_PATH)

mcp = FastMCP(name="Expense Tracker")

def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT DEFAULT '',
                subcategory TEXT DEFAULT '',
                note TEXT DEFAULT ''
            );
        """)

init_db()

@mcp.tool
def add_expense(date: str, amount: float, category: str = "", subcategory: str = "", note: str = "") -> None:
    """Add a new expense to the database."""
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            "INSERT INTO expenses (date, amount, category, subcategory, note) VALUES (?, ?, ?, ?, ?)", 
                                        (date, amount, category, subcategory, note)
            )
        return {"status": "success", "id": cur.lastrowid}

@mcp.tool
def list_expense(start_date: str = "", end_date: str = ""):
    """List all expenses from the database ordered by date."""
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            """
                SELECT 
                    id, date, amount, category, subcategory, note 
                FROM expenses 
                    WHERE date BETWEEN ? AND ?
                ORDER BY date DESC
            """,
            (start_date, end_date)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

@mcp.tool
def summarize(start_date, end_date, category=None):
    '''Summarize expenses by category within an inclusive date range.'''
    query = (
        """
            SELECT category, SUM(amount) AS total_amount
            FROM expenses
            WHERE date BETWEEN ? AND ?
        """
    )
    params = [start_date, end_date]
    
    if category:
        query += " AND category = ?"
        params.append(category)
    
    query += " GROUP BY category ORDER BY total_amount DESC"
    
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(query, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

@mcp.resource("expense://categories", mime_type="application/json")
def categories():
    # Read fresh each time so you can edit the file without restarting
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    mcp.run()