from datetime import datetime

import requests
from bs4 import BeautifulSoup

from app.core.config import get_settings
from app.schemas import ScrapedEvent, ScrapedFight, ScrapedFighter


class UFCStatsScraper:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 UFC analytics research bot; contact: local-development",
            }
        )

    def get_upcoming_events(self) -> list[ScrapedEvent]:
        page = self._get_soup(f"{self.settings.ufc_stats_base_url}/statistics/events/upcoming?page=all")
        rows = page.select("tr.b-statistics__table-row")
        events: list[ScrapedEvent] = []

        for row in rows:
            link = row.select_one("a.b-link_style_black")
            if link is None:
                continue

            cells = [cell.get_text(" ", strip=True) for cell in row.select("td")]
            event_name = link.get_text(" ", strip=True)
            event_url = link.get("href")
            event_date = self._parse_date(cells[0]) if cells else None
            location = cells[1] if len(cells) > 1 else None
            fights = self.get_event_fights(event_url) if event_url else []

            events.append(
                ScrapedEvent(
                    name=event_name,
                    event_date=event_date,
                    location=location,
                    source_url=event_url,
                    fights=fights,
                )
            )

        return events

    def get_event_fights(self, event_url: str) -> list[ScrapedFight]:
        page = self._get_soup(event_url)
        fights: list[ScrapedFight] = []

        for row in page.select("tr.b-fight-details__table-row"):
            fighter_links = row.select("a.b-link_style_black")
            fighter_names = [link.get_text(" ", strip=True) for link in fighter_links if link.get_text(strip=True)]
            if len(fighter_names) < 2:
                continue

            profile_urls = [link.get("href") for link in fighter_links]
            details = [cell.get_text(" ", strip=True) for cell in row.select("td")]
            weight_class = details[6] if len(details) > 6 else None
            bout_type = details[7] if len(details) > 7 else None
            fight_url = row.get("data-link")

            red = self.get_fighter_profile(fighter_names[0], profile_urls[0] if profile_urls else None)
            blue = self.get_fighter_profile(fighter_names[1], profile_urls[1] if len(profile_urls) > 1 else None)

            fights.append(
                ScrapedFight(
                    red_fighter=red,
                    blue_fighter=blue,
                    weight_class=weight_class,
                    bout_type=bout_type,
                    source_url=fight_url,
                )
            )

        return fights

    def get_fighter_profile(self, name: str, profile_url: str | None) -> ScrapedFighter:
        if not profile_url:
            return ScrapedFighter(name=name)

        page = self._get_soup(profile_url)
        record_text = page.select_one(".b-content__title-record")
        record = self._parse_record(record_text.get_text(" ", strip=True) if record_text else "")
        facts = self._parse_fighter_facts(page)

        return ScrapedFighter(
            name=name,
            profile_url=profile_url,
            height=facts.get("HEIGHT"),
            weight=facts.get("WEIGHT"),
            reach=facts.get("REACH"),
            stance=facts.get("STANCE"),
            dob=facts.get("DOB"),
            wins=record[0],
            losses=record[1],
            draws=record[2],
        )

    def _get_soup(self, url: str) -> BeautifulSoup:
        response = self.session.get(url, timeout=20)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _parse_date(self, value: str):
        value = value.strip()
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        return None

    def _parse_record(self, value: str) -> tuple[int | None, int | None, int | None]:
        value = value.replace("Record:", "").strip()
        parts = value.split("-")
        if len(parts) < 3:
            return None, None, None
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return None, None, None

    def _parse_fighter_facts(self, page: BeautifulSoup) -> dict[str, str]:
        facts: dict[str, str] = {}
        for item in page.select(".b-list__box-list-item"):
            label = item.select_one(".b-list__box-item-title")
            if label is None:
                continue
            key = label.get_text(" ", strip=True).replace(":", "").upper()
            value = item.get_text(" ", strip=True).replace(label.get_text(" ", strip=True), "").strip()
            if value and value != "--":
                facts[key] = value
        return facts

