#!/usr/bin/env python3
"""pack_a_punch: email the newest weekly packet to your phone -- on any OS.

Finds the most recently written packet HTML under data/results/ (two-sided
packet_shared/*/packet_*.html or a neural .../packet.html) and sends it as an
attachment through Gmail's SMTP server -- sent from SENDER, delivered to
RECIPIENT -- so it's one tap away in the Gmail app on your phone. The
headline table's picture (the packet's own "Copy" image) goes in the email
body too, so the picks are readable without opening anything. Standard
library only, so it runs anywhere Python 3 does:

    Mac / Linux / WSL:  ./pack_a_punch.py  (or an alias pointing at it)
    Windows:            a PowerShell function running it through WSL's
                        Python -- Windows has no Python of its own here, and
                        the packets live in the WSL checkout anyway.

Gmail needs an app password for SENDER's account, not its normal password
(signed in as SENDER: Google account -> Security -> 2-Step Verification ->
App passwords). Keep it out of the repo, once per machine: in
~/.config/pack_a_punch/gmail_app_password or in the PACK_A_PUNCH_APP_PASSWORD
environment variable.

    pack_a_punch             # send the newest packet
    pack_a_punch --dry-run   # show what would be sent, send nothing
"""
import argparse
import base64
import os
import re
import smtplib
import ssl
import sys
from datetime import datetime
from email.message import EmailMessage
from email.utils import make_msgid
from html import escape
from pathlib import Path

SENDER = 'beniboo2@gmail.com'  # Logs in and sends; the app password belongs to this account.
RECIPIENT = 'jbenreichert@gmail.com'
RESULTS = Path(__file__).resolve().parent / 'data/results'
PASSWORD_FILE = Path.home() / '.config/pack_a_punch/gmail_app_password'
# Gmail caps a whole message at 25 MB, and base64 encoding turns every 3
# bytes of attachment into 4 -- so the file itself has to stay under ~18 MB.
GMAIL_LIMIT = 25_000_000


def newest_packet():
    candidates = list(RESULTS.glob('packet_shared/*/packet_*.html')) + list(RESULTS.glob('*/packet*/*/packet.html'))
    if not candidates:
        raise SystemExit(f'No packet found under {RESULTS} -- run weekly_packet.py first.')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def describe(packet):
    """'2026 Week 2' from a results folder like 2026_2_20 (season_week_lookback)."""
    found = re.search(r'(\d{4})_(\d{1,2})_\d+', str(packet.relative_to(RESULTS)))
    return f'{found.group(1)} Week {int(found.group(2))}' if found else packet.stem


def app_password():
    password = os.environ.get('PACK_A_PUNCH_APP_PASSWORD')
    if not password and PASSWORD_FILE.exists():
        password = PASSWORD_FILE.read_text().strip()
    if not password:
        raise SystemExit(f'No Gmail app password for {SENDER}: put it in {PASSWORD_FILE} (chmod 600) '
                         'or set PACK_A_PUNCH_APP_PASSWORD.')
    return password.replace(' ', '')  # Google shows it in groups of four; spaces aren't part of it.


def headline_picture(html):
    """(PNG bytes, display width) of the picks table the packet embeds for
    its Copy button (weekly_packet.picks_png), or None for a packet built
    before that existed."""
    found = re.search(r'<img id="picks-png"[^>]*?width="(\d+)"[^>]*?src="data:image/png;base64,([^"]+)"', html)
    return (base64.b64decode(found.group(2)), int(found.group(1))) if found else None


def build_message(packet, to):
    written = datetime.fromtimestamp(packet.stat().st_mtime)
    data = packet.read_bytes()
    note = f'{packet.name} ({len(data) / 1e6:.1f} MB), generated {written:%a %b %d, %I:%M %p}.'
    message = EmailMessage()
    message['Subject'] = f'NFL packet · {describe(packet)}'
    message['From'] = SENDER
    message['To'] = to
    message.set_content(f'{note}\nFrom {packet.relative_to(RESULTS.parent.parent)}\n')
    picture = headline_picture(data.decode('utf-8', errors='replace'))
    if picture:
        # HTML body with the table inline (a related cid: image, which Gmail
        # shows in place); the plain text above stays as the fallback.
        png, width = picture
        cid = make_msgid(domain='pack-a-punch')
        message.add_alternative(
            f'<p style="margin:0 0 12px"><img src="cid:{cid[1:-1]}" width="{width}" '
            f'alt="Model picks, {escape(describe(packet))}" style="max-width:100%;height:auto;display:block"></p>'
            f'<p style="font:13px Arial,sans-serif;color:#555;margin:0">{escape(note)} Full packet attached.</p>',
            subtype='html')
        message.get_payload()[1].add_related(png, 'image', 'png', cid=cid, disposition='inline',
                                               filename=f'picks_{packet.stem}.png')
    message.add_attachment(data, maintype='text', subtype='html', filename=packet.name)
    return message


def main(argv=None):
    parser = argparse.ArgumentParser(description='Email the newest weekly packet.')
    parser.add_argument('--to', default=RECIPIENT)
    parser.add_argument('--dry-run', action='store_true', help='Find and package the packet, but send nothing')
    args = parser.parse_args(argv)
    packet = newest_packet()
    message = build_message(packet, args.to)
    encoded = len(message.as_bytes())
    inline = 'headline table in the body' if message.is_multipart() and any(
        part.get_content_type() == 'image/png' for part in message.walk()) else 'no headline picture in this packet'
    print(f'{packet.relative_to(RESULTS.parent.parent)}  ->  {args.to}  (from {SENDER})\n'
          f'  subject "{message["Subject"]}", {inline}, {encoded / 1e6:.1f} MB as an email (Gmail max {GMAIL_LIMIT / 1e6:.0f})')
    if encoded > GMAIL_LIMIT:
        raise SystemExit('Too big for Gmail -- not sent.')
    if args.dry_run:
        print('Dry run: nothing sent.')
        return
    # Verified TLS: this connection carries the app password.
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=120, context=ssl.create_default_context()) as server:
        server.login(SENDER, app_password())
        server.send_message(message)
    print('Sent.')


if __name__ == '__main__':
    sys.exit(main())
