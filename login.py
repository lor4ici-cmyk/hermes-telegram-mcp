#!/usr/bin/env python3
"""Interactive QR login; run only in your own terminal."""
import asyncio
import getpass

import qrcode
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError

from server import SESSION, credentials


def show_qr(url):
    qr = qrcode.QRCode(border=2)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


async def main():
    SESSION.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    api_id, api_hash = credentials()
    client = TelegramClient(str(SESSION), api_id, api_hash)
    try:
        await client.connect()
        if not await client.is_user_authorized():
            print("Telegram → Настройки → Устройства → Подключить устройство; сканируйте QR ниже.", flush=True)
            while not await client.is_user_authorized():
                qr_login = await client.qr_login()
                show_qr(qr_login.url)
                print("Ожидаю сканирования (QR действует около 30 секунд)...", flush=True)
                try:
                    await qr_login.wait(timeout=30)
                except asyncio.TimeoutError:
                    print("QR устарел, обновляю...", flush=True)
                    continue
                except SessionPasswordNeededError:
                    password = getpass.getpass("Пароль Telegram 2FA: ")
                    await client.sign_in(password=password)
                    del password
        if not await client.is_user_authorized():
            raise RuntimeError("Авторизация Telegram не завершена")
        print("Telegram авторизован. Перезапустите Hermes CLI.")
    finally:
        await client.disconnect()
        if SESSION.exists():
            SESSION.chmod(0o600)


if __name__ == "__main__":
    asyncio.run(main())
