"""
Task 7: SQL Data Analysis with SQLite
CoreTech Client Dataset Analysis
"""

import pandas as pd
import sqlite3
import os

# ─────────────────────────────────────────────
# STEP 1: Load CoreTech Client Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("TASK 7: SQL Data Analysis with SQLite")
print("=" * 60)

csv_path = "coretech_clients_cleaned.csv"   # update path if needed in Colab

df = pd.read_csv(csv_path)
print(f"\n✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# STEP 2: Create SQLite Database
# ─────────────────────────────────────────────
db_path = "coretech_clients.db"

if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
print(f"✅ SQLite database created: {db_path}")

# ─────────────────────────────────────────────
# STEP 3: Create 'clients' Table
# ─────────────────────────────────────────────
cursor.execute("""
    CREATE TABLE IF NOT EXISTS clients (
        client_id            TEXT PRIMARY KEY,
        client_name          TEXT,
        company_name         TEXT,
        service              TEXT,
        project_budget       REAL,
        project_duration_days INTEGER,
        client_city          TEXT,
        lead_source          TEXT,
        status               TEXT,
        rating               REAL
    )
""")
conn.commit()
print("✅ Table 'clients' created\n")

# ─────────────────────────────────────────────
# STEP 4: Insert Data from CSV
# ─────────────────────────────────────────────
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO clients VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row['client_id'],
        row['client_name'],
        row['company_name'],
        row['service'],
        row['project_budget'],
        row['project_duration_days'],
        row['client_city'],
        row['lead_source'],
        row['status'],
        row['rating'],
    ))

conn.commit()
print(f"✅ {df.shape[0]} records inserted into 'clients' table\n")

# ─────────────────────────────────────────────
# Helper: Run & Display a Query
# ─────────────────────────────────────────────
def run_query(title, query, explanation):
    print("─" * 60)
    print(f"📌 {title}")
    print(f"💡 Explanation: {explanation}")
    print(f"\n🔍 SQL:\n{query.strip()}\n")
    result = pd.read_sql_query(query, conn)
    print(result.to_string(index=False))
    print()

# ─────────────────────────────────────────────
# STEP 5: 10 SQL Queries
# ─────────────────────────────────────────────

# Query 1: Show All Clients
run_query(
    title="Query 1: Show All Clients",
    query="SELECT * FROM clients;",
    explanation="Retrieves every record in the clients table — gives a complete overview of the dataset."
)

# Query 2: Filter by City
run_query(
    title="Query 2: Filter Clients by City (New York)",
    query="""
SELECT client_name, company_name, service, project_budget
FROM clients
WHERE client_city = 'New York';
""",
    explanation="Filters and shows only clients located in New York. Useful for city-level business analysis."
)

# Query 3: Show Completed Projects
run_query(
    title="Query 3: Show Completed Projects",
    query="""
SELECT client_name, company_name, service, status
FROM clients
WHERE status = 'Completed';
""",
    explanation="Lists all clients whose project status is 'Completed' to track successful deliveries."
)

# Query 4: Count Clients by Service
run_query(
    title="Query 4: Count Clients by Service",
    query="""
SELECT service, COUNT(*) AS total_clients
FROM clients
GROUP BY service
ORDER BY total_clients DESC;
""",
    explanation="Groups clients by service type and counts each group — shows which service is most in demand."
)

# Query 5: Average Budget by Service
run_query(
    title="Query 5: Average Budget by Service",
    query="""
SELECT service,
       ROUND(AVG(project_budget), 2) AS avg_budget
FROM clients
GROUP BY service
ORDER BY avg_budget DESC;
""",
    explanation="Calculates the average project budget per service type to understand revenue contribution of each."
)

# Query 6: Highest, Lowest, and Average Budget
run_query(
    title="Query 6: Highest, Lowest, and Average Budget",
    query="""
SELECT
    MAX(project_budget) AS highest_budget,
    MIN(project_budget) AS lowest_budget,
    ROUND(AVG(project_budget), 2) AS average_budget
FROM clients;
""",
    explanation="Returns the budget extremes and overall average — useful for understanding the financial range of projects."
)

# Query 7: Clients with Rating Above 4
run_query(
    title="Query 7: Clients with Rating Above 4",
    query="""
SELECT client_name, company_name, service, rating
FROM clients
WHERE rating > 4
ORDER BY rating DESC;
""",
    explanation="Filters highly-satisfied clients with a rating above 4 to identify top-performing projects."
)

# Query 8: Count Clients by Status
run_query(
    title="Query 8: Count Clients by Status",
    query="""
SELECT status, COUNT(*) AS total
FROM clients
GROUP BY status
ORDER BY total DESC;
""",
    explanation="Shows the distribution of project statuses (Completed, In Progress, Pending) across all clients."
)

# Query 9: Top 5 Highest Budget Projects
run_query(
    title="Query 9: Top 5 Highest Budget Projects",
    query="""
SELECT client_name, company_name, service, project_budget, client_city
FROM clients
ORDER BY project_budget DESC
LIMIT 5;
""",
    explanation="Retrieves the 5 most expensive projects — helps identify the highest-value clients and services."
)

# Query 10: Clients per City with Average Rating
run_query(
    title="Query 10: Clients per City with Average Rating",
    query="""
SELECT client_city,
       COUNT(*) AS total_clients,
       ROUND(AVG(project_budget), 2) AS avg_budget,
       ROUND(AVG(rating), 2) AS avg_rating
FROM clients
GROUP BY client_city
ORDER BY total_clients DESC;
""",
    explanation="Summarizes each city's client count, average budget, and satisfaction rating for regional analysis."
)

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
conn.close()
print("=" * 60)
print("✅ All 10 queries completed successfully!")
print(f"✅ Database saved as: {db_path}")
print("=" * 60)
