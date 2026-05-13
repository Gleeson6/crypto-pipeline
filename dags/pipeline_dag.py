"""
pipeline_dag.py — Crypto Pipeline Airflow DAG
----------------------------------------------
This file defines the workflow that Airflow manages.

A DAG (Directed Acyclic Graph) is just a set of tasks with a defined order.
"Directed" = tasks flow in one direction (no going backwards)
"Acyclic"  = no loops (task A can't depend on task B if B depends on A)
"Graph"    = tasks are nodes, dependencies are edges

Our DAG has 3 tasks in sequence:
  start_ingestion → start_consumer → start_trade_engine

Airflow reads this file automatically from the /dags folder.
Any change you make here is picked up within 30 seconds by the scheduler.

Schedule: @once — we trigger it manually from the UI.
          Change to "*/5 * * * *" for every 5 minutes, "0 9 * * *" for 9am daily etc.
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# --- Default arguments applied to every task in this DAG ---
# retries=2        → if a task fails, Airflow retries it 2 times before giving up
# retry_delay      → waits 30 seconds between each retry attempt
# These protect against temporary failures (network blip, container restart etc.)
default_args = {
    'owner': 'gleezon',
    'retries': 2,
    'retry_delay': timedelta(seconds=30),
}

# --- Define the DAG ---
# dag_id       → unique name, shown in the Airflow UI
# start_date   → Airflow needs a start date to calculate run schedules
# schedule     → @once means it only runs when you manually trigger it
# catchup=False → don't backfill missed runs if the start_date was in the past
with DAG(
    dag_id='crypto_pipeline',
    default_args=default_args,
    description='Binance ingestion → Kafka consumer → RSI trade engine',
    start_date=datetime(2026, 1, 1),
    schedule='@once',
    catchup=False,
) as dag:

    # --- Task 1: Start the Binance WebSocket ingestion ---
    # BashOperator runs a shell command as a task.
    # This starts binance_stream.py in the background (&) so Airflow
    # doesn't wait for it to finish (it runs forever by design).
    # The script path uses the mounted project directory in the container.
    start_ingestion = BashOperator(
        task_id='start_ingestion',
        bash_command="""
            cd ~/crypto-pipeline &&
            source venv/bin/activate &&
            nohup python3 ingestion/binance_stream.py > /tmp/ingestion.log 2>&1 &
            echo "Ingestion started — PID: $!"
            sleep 5
        """,
    )

    # --- Task 2: Start the Kafka consumer ---
    # Only runs after start_ingestion succeeds.
    # nohup + & runs it in the background so this task completes
    # while the consumer keeps running independently.
    start_consumer = BashOperator(
        task_id='start_consumer',
        bash_command="""
            cd ~/crypto-pipeline &&
            source venv/bin/activate &&
            nohup python3 storage/consumer.py > /tmp/consumer.log 2>&1 &
            echo "Consumer started — PID: $!"
            sleep 5
        """,
    )

    # --- Task 3: Start the trade engine ---
    # Only runs after start_consumer succeeds.
    # By this point ingestion is flowing into Kafka and consumer
    # is writing to TimescaleDB — the trade engine has data to work with.
    start_trade_engine = BashOperator(
        task_id='start_trade_engine',
        bash_command="""
            cd ~/crypto-pipeline &&
            source venv/bin/activate &&
            nohup python3 strategy/trade_engine.py > /tmp/trade_engine.log 2>&1 &
            echo "Trade engine started — PID: $!"
        """,
    )

    # --- Define the order of execution ---
    # This single line says: ingestion first, then consumer, then trade engine.
    # >> is Airflow's syntax for "then run".
    start_ingestion >> start_consumer >> start_trade_engine
