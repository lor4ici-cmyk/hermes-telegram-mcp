import unittest
from unittest.mock import AsyncMock, patch

import server


class ToolSmokeTests(unittest.TestCase):
    def test_tools_are_registered(self):
        expected = {
            "unread_digest", "find_chat", "recent_messages", "search_messages",
            "download_attachment", "send_message", "send_file", "edit_message",
            "delete_message", "send_temporary_message", "preview_telegram_chat", "join_telegram_chat",
        }
        self.assertTrue(expected.issubset(set(server.mcp._tool_manager._tools)))

    def test_send_message_calls_mocked_client(self):
        client = AsyncMock()
        client.get_entity.return_value = 123456
        sent = type("Message", (), {"id": 42})()
        client.send_message.return_value = sent
        import asyncio
        with patch.object(server, "_client_get", new=AsyncMock(return_value=client)):
            result = asyncio.run(server.send_message(123456, "simulation only"))
        self.assertIn('"ok": true', result)
        self.assertEqual(client.send_message.await_count, 1)

    def test_delete_verification_uses_mock(self):
        client = AsyncMock()
        entity = object()
        message = type("Message", (), {"out": True})()
        client.get_messages.return_value = message
        client.delete_messages.return_value = None
        import asyncio
        result = asyncio.run(server._delete_verified(client, entity, 123456, 42))
        self.assertTrue(result["deleted"])
        client.delete_messages.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
