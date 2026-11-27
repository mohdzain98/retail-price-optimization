"""Static artwork of the sign-in page: the story column and the sign-in card header.

Built as HTML because Streamlit has no native icon tiles, underline swash or icon
stat strip. Icons are SVG data-URI images (st.html strips inline <svg>). Colours
follow the active light or dark theme; interactive parts stay native widgets.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

ICONS: dict[str, str] = {
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/>'
    '<polyline points="16 7 22 7 22 13"/>',
    "store": '<path d="M3 9.5 5 4h14l2 5.5"/><path d="M4 10v10h16V10"/><path d="M9 20v-5h6v5"/>'
    '<path d="M3 9.5a3 3 0 0 0 6 0 3 3 0 0 0 6 0 3 3 0 0 0 6 0"/>',
    "tag": '<path d="M12.6 2.6A2 2 0 0 0 11.2 2H4a2 2 0 0 0-2 2v7.2a2 2 0 0 0 .6 1.4l8.7 8.7'
    'a2.4 2.4 0 0 0 3.4 0l6.6-6.6a2.4 2.4 0 0 0 0-3.4z"/><circle cx="7.5" cy="7.5" r="1"/>',
    "calendar": '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/>'
    '<path d="m9 16 2 2 4-4"/>',
    "package": '<path d="M11 21.7a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16V8a2 2 0 0 0-1-1.7l-7-4a2 2 0 0 '
    '0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.7z"/><path d="M12 22V12"/>'
    '<path d="m3.3 7 8.7 5 8.7-5"/>',
    "layers": '<path d="m12.8 2.2a2 2 0 0 0-1.6 0L2.6 6.1a1 1 0 0 0 0 1.8l8.6 3.9a2 2 0 0 0 1.6 '
    '0l8.6-3.9a1 1 0 0 0 0-1.8z"/><path d="m22 17.6-9.2 4.2a2 2 0 0 1-1.6 0L2 17.6"/>'
    '<path d="m22 12.6-9.2 4.2a2 2 0 0 1-1.6 0L2 12.6"/>',
    "chart": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M8 17v-3M13 17V9M18 17V5"/>',
    "login": '<path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/><path d="m10 17 5-5-5-5"/>'
    '<path d="M15 12H3"/>',
}
POINTS: tuple[tuple[str, str, str], ...] = (
    ("trend", "Measure", "how each product's demand reacts to price and promotions."),
    ("tag", "Optimize", "regular prices within your own guardrails and margins."),
    ("calendar", "Plan", "a 13-week promo calendar and the budget it deserves."),
)
STATS: tuple[tuple[str, str], ...] = (
    ("package", "54 products"),
    ("layers", "4 categories"),
    ("chart", "3 years of weekly store sales"),
)


@dataclass(frozen=True)
class Palette:
    """Colours of the sign-in artwork for one appearance (light or dark)."""

    ink: str
    muted: str
    accent: str
    tile: str
    chip: str
    rule: str

    @classmethod
    def for_theme(cls, theme_type: str | None) -> Palette:
        """Dark palette when the active theme is dark, else light."""
        if theme_type == "dark":
            return cls("#F1F5F9", "#94A3B8", "#60A5FA", "rgba(96,165,250,0.16)", "#1E293B",
                       "#334155")  # fmt: skip
        return cls("#0F172A", "#475569", "#1D5CD6", "#EAF1FC", "#F1F5F9", "#CBD5E1")


class LandingArt:
    """HTML for the story column and the sign-in card header.

    Every element is static; all sizes are in px so it renders the same everywhere.
    """

    def __init__(self, palette: Palette) -> None:
        self._p = palette

    def story(self) -> str:
        """Badges, three-line headline with an underline swash, points and the stat strip."""
        p = self._p
        badges = self._chip("trend", "Revenue Growth Management", p.accent, p.tile) + self._chip(
            "store", "dunnhumby retail data", p.ink, p.chip
        )
        points = "".join(
            f'<div style="display:flex;align-items:center;gap:16px;margin:0 0 12px">'
            f"{self._tile(icon, 36, 18)}"
            f'<div style="font-size:15px;color:{p.ink};line-height:1.45">'
            f"<b>{verb}</b> {rest}</div></div>"
            for icon, verb, rest in POINTS
        )
        stats = f'<span style="color:{p.rule};margin:0 12px">|</span>'.join(
            f'<span style="display:inline-flex;align-items:center;gap:8px">'
            f"{self._svg(icon, 16, p.muted)}{text}</span>"
            for icon, text in STATS
        )
        return (
            f'<div style="font-family:inherit">'
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:24px">{badges}</div>'
            f'<div style="font-size:40px;font-weight:800;line-height:48.2px;letter-spacing:-0.02em;'
            f'color:{p.ink};margin-bottom:28px">Price smarter.<br>'
            f'<span style="color:{p.accent};position:relative;display:inline-block">'
            f"Promote with purpose.{self._swash()}</span><br>Plan with proof.</div>"
            f"{points}"
            f'<div style="display:flex;align-items:center;flex-wrap:wrap;margin-top:22px;'
            f'font-size:14px;color:{p.muted}">{stats}</div></div>'
        )

    def card_header(self) -> str:
        """Icon tile plus the "Sign in" title."""
        return (
            f'<div style="display:flex;align-items:center;gap:14px">{self._tile("login", 40, 20)}'
            f'<span style="font-size:22px;font-weight:700;color:{self._p.ink}">Sign in</span></div>'
        )

    def _chip(self, icon: str, text: str, color: str, background: str) -> str:
        """Rounded badge with an icon."""
        return (
            f'<span style="display:inline-flex;align-items:center;gap:7px;padding:6px 12px;'
            f'border-radius:10px;background:{background};color:{color};font-size:13px">'
            f"{self._svg(icon, 15, color)}{text}</span>"
        )

    def _tile(self, icon: str, size: int, glyph: int) -> str:
        """Rounded square with a centred accent icon."""
        return (
            f'<span style="flex:none;display:inline-flex;align-items:center;justify-content:center;'
            f'width:{size}px;height:{size}px;border-radius:10px;background:{self._p.tile}">'
            f"{self._svg(icon, glyph, self._p.accent)}</span>"
        )

    def _swash(self) -> str:
        """Light hand-drawn underline under the accent line."""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 14">'
            f'<path d="M3 11 C 40 3, 100 2, 137 6" fill="none" stroke="{self._p.accent}" '
            'stroke-width="3" stroke-linecap="round" opacity="0.55"/></svg>'
        )
        return (
            f'<img src="{self._data_uri(svg)}" alt="" style="position:absolute;right:2px;'
            'bottom:-10px;width:110px;height:11px">'
        )

    @classmethod
    def _svg(cls, icon: str, size: int, color: str) -> str:
        """Lucide-style stroke icon as an <img> (inline <svg> is stripped by st.html)."""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" '
            f'stroke-linejoin="round">{ICONS[icon]}</svg>'
        )
        return f'<img src="{cls._data_uri(svg)}" alt="" width="{size}" height="{size}">'

    @staticmethod
    def _data_uri(svg: str) -> str:
        """Base64 data URI of an SVG document."""
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
