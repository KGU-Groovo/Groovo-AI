from app.tunnel import TUNNEL_URL_PATTERN, get_websocket_url


def test_converts_quick_tunnel_url_to_websocket_url():
    assert get_websocket_url("https://dancing-bear.trycloudflare.com") == "wss://dancing-bear.trycloudflare.com/ws/analyze"


def test_finds_quick_tunnel_url_in_cloudflared_output():
    output = "INF +------------------------------------------------------------+\nINF |  https://dancing-bear.trycloudflare.com                 |\n"
    assert TUNNEL_URL_PATTERN.search(output).group(0) == "https://dancing-bear.trycloudflare.com"
