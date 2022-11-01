"""The database manager for marianne"""
# marianne/db.py

import sqlite3

import click
from flask import current_app, g


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(
            current_app.config["METADATA_DATABASE"],
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop("db", None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()

    with current_app.open_resource("schema/metadata.sql") as f:
        db.executescript(f.read().decode("utf8"))


def insert_metadata(metadata):
    db = get_db()

    try:
        db.executemany(
            "INSERT OR REPLACE INTO METADATA (title, url, desc) values(?, ?, ?)",
            [metadata],
        )
    except Exception as e:
        print("[!] Error updating database ->", e)

    db.commit()
    db.close()


def select_all_metadata():
    db = get_db()

    try:
        rows = db.execute(
            "SELECT * FROM METADATA ORDER BY title",
        )
        db.close()
        return rows
    except Exception as e:
        print("[!] Error select data from database ->", e)
        return []


@click.command("init-db")
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo("Initialized the database.")


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
