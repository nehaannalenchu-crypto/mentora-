"""Add clearly labeled fictional tutor accounts for a local presentation demo."""
import json
from app import ROOT, db, now_iso


def seed_demo_tutors():
    profiles = json.loads((ROOT / 'data' / 'demo_tutors.json').read_text())
    with db() as connection:
        for profile in profiles:
            email = f"demo.{profile['demo_slug']}@mentora.invalid"
            connection.execute('INSERT OR IGNORE INTO users(email,role,created_at) VALUES(?,?,?)', (email, 'teacher', now_iso()))
            user_id = connection.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()[0]
            connection.execute('''INSERT INTO profiles(user_id,profile_json,complete,verification_status,updated_at)
                VALUES(?,?,1,'approved',?) ON CONFLICT(user_id) DO UPDATE SET
                profile_json=excluded.profile_json,complete=1,verification_status='approved',updated_at=excluded.updated_at''',
                (user_id, json.dumps(profile), now_iso()))
    print(f'Added or updated {len(profiles)} fictional demo tutor profiles.')


if __name__ == '__main__':
    seed_demo_tutors()
