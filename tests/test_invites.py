import unittest
from unittest.mock import AsyncMock, patch

import server
from telethon import TelegramClient
from telethon.errors import InviteHashExpiredError, InviteHashInvalidError, InviteRequestSentError, UserAlreadyParticipantError


class InviteParsingTests(unittest.TestCase):
    def test_public_handle_and_post_link(self):
        self.assertEqual(server._parse_telegram_target("@example"), ("username", "example", None))
        self.assertEqual(server._parse_telegram_target("https://t.me/example/123"), ("username", "example", 123))

    def test_private_invite_formats(self):
        self.assertEqual(server._parse_telegram_target("https://t.me/+AbC_12"), ("invite", "AbC_12", None))
        self.assertEqual(server._parse_telegram_target("t.me/joinchat/AbC-12"), ("invite", "AbC-12", None))

    def test_rejects_non_telegram_and_ambiguous_path(self):
        for value in ("https://evil.example/a", "t.me/s", "not a link"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                server._parse_telegram_target(value)


class InviteFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_join_request_sent_is_reported_without_retry(self):
        client = TelegramClient.__new__(TelegramClient)
        client.get_entity = AsyncMock(return_value=object())
        client.get_me = AsyncMock(return_value=type("Me", (), {"id": 77})())
        request = InviteRequestSentError(request=None)
        async def run(op):
            try:
                result = await op(client)
            except Exception as exc:
                result = server._join_error_result(exc)
            if result.get("status") == "joined":
                return {"ok": True, "result": result}
            return {"ok": False, "error": result.get("status", "telegram_rejected"), **result}
        with patch.object(server, "_get_membership", new=AsyncMock(return_value="not_member")), patch.object(server, "JoinChannelRequest", return_value=object()), patch.object(server, "_client_get", new=AsyncMock(return_value=client)), patch.object(server, "_run", new=AsyncMock(side_effect=run)):
            with patch("telethon.client.users.UserMethods.__call__", new=AsyncMock(side_effect=request)):
                import json
                result = json.loads(await server.join_telegram_chat("@example"))
        self.assertEqual(result["status"], "request_sent")

    async def test_invalid_invite_is_rejected(self):
        self.assertEqual(server._join_error_result(InviteHashInvalidError(request=None))["status"], "telegram_rejected")

    async def test_already_participant_is_reported(self):
        self.assertEqual(server._join_error_result(UserAlreadyParticipantError(request=None))["status"], "already_member")

    async def test_expired_invite_is_reported(self):
        client = TelegramClient.__new__(TelegramClient)
        request = InviteHashExpiredError(request=None)
        async def run(op):
            try:
                result = await op(client)
            except Exception as exc:
                result = server._join_error_result(exc)
            if result.get("status") == "joined":
                return {"ok": True, "result": result}
            return {"ok": False, "error": result.get("status", "telegram_rejected"), **result}
        with patch.object(server, "_client_get", new=AsyncMock(return_value=client)), patch.object(server, "ImportChatInviteRequest", return_value=object()), patch.object(server, "_run", new=AsyncMock(side_effect=run)):
            with patch("telethon.client.users.UserMethods.__call__", new=AsyncMock(side_effect=request)):
                import json
                result = json.loads(await server.join_telegram_chat("t.me/+Expired"))
        self.assertEqual(result["status"], "invite_expired")


if __name__ == "__main__":
    unittest.main()
