from typing import Optional, Dict, Any
from fastmcp import FastMCP
import os
import aiosqlite  # Changed: sqlite3 → aiosqlite
import tempfile

# Use temporary directory which should be writable on deployment
TEMP_DIR = tempfile.gettempdir()
DB_PATH = os.path.join(TEMP_DIR, "expenses.db")
# DB_PATH = os.path.join(os.path.dirname(__file__), "database", "expense_tracker.db")   // uncomment to use local db
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "database", "categories.json")

print(f"Database path: {DB_PATH}")

mcp = FastMCP("ExpenseTracker")

def init_db():  # Keep as sync for initialization
    try:
        # Use synchronous sqlite3 just for initialization
        import sqlite3
        with sqlite3.connect(DB_PATH) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("""
                CREATE TABLE IF NOT EXISTS expenses(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    amount REAL NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT DEFAULT '',
                    note TEXT DEFAULT ''
                )
            """)
            # Test write access
            c.execute("INSERT OR IGNORE INTO expenses(date, amount, category) VALUES ('2000-01-01', 0, 'test')")
            c.execute("DELETE FROM expenses WHERE category = 'test'")
            print("Database initialized successfully with write access")
    except Exception as e:
        print(f"Database initialization error: {e}")
        raise

# Initialize database synchronously at module load
init_db()

@mcp.tool()
async def add_expense(date, amount, category, subcategory="", note=""):  # Changed: added async
    '''Add a new expense entry to the database.'''
    try:
        async with aiosqlite.connect(DB_PATH) as c:  # Changed: added async
            cur = await c.execute(  # Changed: added await
                "INSERT INTO expenses(date, amount, category, subcategory, note) VALUES (?,?,?,?,?)",
                (date, amount, category, subcategory, note)
            )
            expense_id = cur.lastrowid
            await c.commit()  # Changed: added await
            return {"status": "success", "id": expense_id, "message": "Expense added successfully"}
    except Exception as e:  # Changed: simplified exception handling
        if "readonly" in str(e).lower():
            return {"status": "error", "message": "Database is in read-only mode. Check file permissions."}
        return {"status": "error", "message": f"Database error: {str(e)}"}
    
@mcp.tool()
async def list_expenses(start_date, end_date):  # Changed: added async
    '''List expense entries within an inclusive date range.'''
    try:
        async with aiosqlite.connect(DB_PATH) as c:  # Changed: added async
            cur = await c.execute(  # Changed: added await
                """
                SELECT id, date, amount, category, subcategory, note
                FROM expenses
                WHERE date BETWEEN ? AND ?
                ORDER BY date DESC, id DESC
                """,
                (start_date, end_date)
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in await cur.fetchall()]  # Changed: added await
    except Exception as e:
        return {"status": "error", "message": f"Error listing expenses: {str(e)}"}

@mcp.tool()
async def summarize(start_date, end_date, category=None):  # Changed: added async
    '''Summarize expenses by category within an inclusive date range.'''
    try:
        async with aiosqlite.connect(DB_PATH) as c:  # Changed: added async
            query = """
                SELECT category, SUM(amount) AS total_amount, COUNT(*) as count
                FROM expenses
                WHERE date BETWEEN ? AND ?
            """
            params = [start_date, end_date]

            if category:
                query += " AND category = ?"
                params.append(category)

            query += " GROUP BY category ORDER BY total_amount DESC"

            cur = await c.execute(query, params)  # Changed: added await
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in await cur.fetchall()]  # Changed: added await
    except Exception as e:
        return {"status": "error", "message": f"Error summarizing expenses: {str(e)}"}

@mcp.tool
async def delete_expense(
    id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    amount: Optional[float] = None,
) -> dict:
    """Delete expenses matching the provided filters (combined with AND). At least one filter is required."""
    async with aiosqlite.connect(DB_PATH) as c:
        where_clauses = []
        params = []

        if id is not None:
            where_clauses.append("id = ?")
            params.append(id)

        if start_date:
            where_clauses.append("date >= ?")
            params.append(start_date)

        if end_date:
            where_clauses.append("date <= ?")
            params.append(end_date)

        if category:
            where_clauses.append("category = ?")
            params.append(category)

        if subcategory:
            where_clauses.append("subcategory = ?")
            params.append(subcategory)

        if amount is not None:
            where_clauses.append("amount = ?")
            params.append(amount)

        if not where_clauses:
            raise ValueError("At least one filter must be provided to avoid deleting all expenses.")

        where = " AND ".join(where_clauses)
        query = f"DELETE FROM expenses WHERE {where}"

        cur = await c.execute(query, params)
        await c.commit()
        deleted = cur.rowcount

        return {
            "status": "success" if deleted > 0 else "no_match",
            "deleted_count": deleted
        }

@mcp.tool
async def edit_expense(
# ── Filters (to find which expense(s) to edit) ─────────────────────────────────
    id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    amount: Optional[float] = None,
    note_contains: Optional[str] = None,

# ── Fields to update (only provide the ones you want to change) ───────────────
    new_date: Optional[str] = None,
    new_amount: Optional[float] = None,
    new_category: Optional[str] = None,
    new_subcategory: Optional[str] = None,
    new_note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Edit one or more expenses matching the filters.
    Only the 'new_*' fields you provide will be updated.
    At least one filter is required (prevents accidental mass updates).
    """
    if all(f is None for f in [id, start_date, end_date, category, subcategory, amount, note_contains]):
        raise ValueError("At least one filter must be provided to select which expense(s) to edit.")
    
    # Build WHERE clause (same logic as delete_expense)
    where_clauses = []
    params = []
    
    if id is not None:
        where_clauses.append("id = ?")
        params.append(id)
    if start_date:
        where_clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("date <= ?")
        params.append(end_date)
    if category:
        where_clauses.append("category = ?")
        params.append(category)
    if subcategory:
        where_clauses.append("subcategory = ?")
        params.append(subcategory)
    if amount is not None:
        where_clauses.append("amount = ?")
        params.append(amount)
    if note_contains:
        where_clauses.append("note LIKE ?")
        params.append(f"%{note_contains}%")
        
    where_sql = " AND ".join(where_clauses)
    
    # Build SET clause – only include fields that are not None
    set_clauses = []
    set_params = []
    
    if new_date is not None:
        set_clauses.append("date = ?")
        set_params.append(new_date)
    if new_amount is not None:
        set_clauses.append("amount = ?")
        set_params.append(new_amount)
    if new_category is not None:
        set_clauses.append("category = ?")
        set_params.append(new_category)
    if new_subcategory is not None:
        set_clauses.append("subcategory = ?")
        set_params.append(new_subcategory)
    if new_note is not None:
        set_clauses.append("note = ?")
        set_params.append(new_note)
    
    if not set_clauses:
        raise ValueError("At least one field to update must be provided.")
    
    set_sql = ", ".join(set_clauses)
    query = f"UPDATE expenses SET {set_sql} WHERE {where_sql}"
    all_params = set_params + params
    
    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute(query, all_params)
        await c.commit()
        updated_count = cur.rowcount
        
    status = "success" if updated_count > 0 else "no_match"
    return {
        "status": status,
        "updated_count": updated_count,
        "message": f"Updated {updated_count} expense(s)." if updated_count else "No expenses matched the filters."
    }

@mcp.resource("expense:///categories", mime_type="application/json")  # Changed: expense:// → expense:///
def categories():
    try:
        # Provide default categories if file doesn't exist
        default_categories = {
            "categories": [
                "Food & Dining",
                "Transportation",
                "Shopping",
                "Entertainment",
                "Bills & Utilities",
                "Healthcare",
                "Travel",
                "Education",
                "Business",
                "Other"
            ]
        }
        
        try:
            with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            import json
            return json.dumps(default_categories, indent=2)
    except Exception as e:
        return f'{{"error": "Could not load categories: {str(e)}"}}'

# Start the server
if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
    # mcp.run()
    
    # for running the server: fastmcp run main.py --transport http --host 0.0.0.0 --port 8000 or uv run main.py
    # In debugging mode: uv run fastmcp dev main.py
    
    # For Deployment visit fastmcp.cloud and deploy the server
    
    # For free Claude-desktop trail, use the 02_proxy.py file to run the remote server locally
    #   - uv run fastmcp install claude-desktop 02_proxy.py