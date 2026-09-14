"""Administrator account management.

Deliberately a command-line tool rather than an API route. Creating an administrator
requires filesystem access to the database, so the web application can never be used
to escalate a registered account into an admin.

The password is read with getpass: it is never echoed, never passed as an argument,
and never lands in shell history.

    python scripts/08_admin.py create <username>
    python scripts/08_admin.py passwd <username>
    python scripts/08_admin.py list
    python scripts/08_admin.py delete <username>
"""
from __future__ import annotations

import argparse
import getpass
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elpr.db.app_store import MIN_PASSWORD, AppStore  # noqa: E402


def _read_new_password(minimum: int) -> str:
    """Prompt twice and require agreement. Returns the password, never prints it."""
    for _ in range(3):
        first = getpass.getpass("New password (not shown): ")
        if len(first) < minimum:
            print(f"  Too short. Needs at least {minimum} characters.")
            continue
        second = getpass.getpass("Confirm password: ")
        if first != second:
            print("  The two entries did not match.")
            continue
        return first
    raise SystemExit("Giving up after three attempts.")


def cmd_create(store: AppStore, args: argparse.Namespace) -> None:
    display = args.name or args.username
    print(f"Creating administrator '{args.username}' ({display}).")
    print(f"Password must be at least {MIN_PASSWORD} characters.")
    password = _read_new_password(MIN_PASSWORD)
    user = store.create_admin(args.username, password, display)
    print(f"\nCreated admin '{user.username}' (id {user.id}).")
    print("Sign in at http://localhost:8420 with that username and password.")


def cmd_passwd(store: AppStore, args: argparse.Namespace) -> None:
    accounts = {a["username"]: a for a in store.list_accounts()}
    target = args.username.strip().lower()
    if target not in accounts:
        raise SystemExit(f"No account named '{target}'.")
    print(f"Resetting password for '{target}' (role {accounts[target]['role']}).")
    password = _read_new_password(MIN_PASSWORD)
    if store.set_password(target, password):
        print(f"\nPassword changed. Any open session for '{target}' was signed out.")


def cmd_list(store: AppStore, args: argparse.Namespace) -> None:
    accounts = store.list_accounts()
    if not accounts:
        print("No accounts registered.")
        return
    print(f"{'USERNAME':<20} {'ROLE':<9} {'MODULE':<8} {'EVENTS':>6}  CREATED")
    for a in accounts:
        created = datetime.fromtimestamp(a["created_at"]).strftime("%Y-%m-%d %H:%M")
        print(f"{a['username']:<20} {a['role']:<9} {a['module'] or '':<8} "
              f"{a['n_events']:>6}  {created}")
    s = store.stats()
    print(f"\n{s['n_users']} accounts: {s['n_students']} student, "
          f"{s['n_advisers']} adviser, {s['n_admins']} admin. "
          f"{s['n_events']} study events, {s['n_recommendations']} recommendations.")


def cmd_delete(store: AppStore, args: argparse.Namespace) -> None:
    target = args.username.strip().lower()
    accounts = {a["username"]: a for a in store.list_accounts()}
    if target not in accounts:
        raise SystemExit(f"No account named '{target}'.")
    a = accounts[target]
    print(f"About to delete '{target}' (role {a['role']}, {a['n_events']} study events).")
    print("This also removes their study history and cannot be undone.")
    if input("Type the username again to confirm: ").strip().lower() != target:
        raise SystemExit("Not confirmed. Nothing was deleted.")
    store.delete_account(target)
    print(f"Deleted '{target}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="create an administrator account")
    p.add_argument("username")
    p.add_argument("--name", help="display name, defaults to the username")
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("passwd", help="reset the password of any account")
    p.add_argument("username")
    p.set_defaults(func=cmd_passwd)

    p = sub.add_parser("list", help="list every registered account")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("delete", help="delete an account and its history")
    p.add_argument("username")
    p.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    store = AppStore()
    try:
        args.func(store, args)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}")


if __name__ == "__main__":
    main()
