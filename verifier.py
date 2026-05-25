import os
import re
import json
import logging
from typing import Optional, List
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class Verifier:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in environment / .env file.")
        self.groq_client = Groq(api_key=api_key)
        self.session = requests.Session()

        self.trusted_sources = [
            "thehindu.com", "ndtv.com", "indiatoday.in", "indianexpress.com",
            "hindustantimes.com", "timesofindia.indiatimes.com", "news18.com",
            "altnews.in", "boomlive.in", "pib.gov.in", "vishvasnews.com",
            "reuters.com", "apnews.com", "bbc.com", "snopes.com", "politifact.com",
            "factcheck.org", "theprint.in", "scroll.in", "thewire.in"
        ]

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    # ─────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────

    def is_trusted(self, url: str) -> bool:
        return any(trusted in url.lower() for trusted in self.trusted_sources)

    def validate_claim(self, claim: str) -> Optional[str]:
        """Returns an error string if claim is invalid, else None."""
        stripped = claim.strip() if claim else ""
        if not stripped:
            return "Claim cannot be empty."
        if len(stripped) < 10:
            return "Claim is too short to fact-check meaningfully."
        if len(stripped) > 1000:
            return "Claim is too long. Please shorten it to under 1000 characters."
        return None

    # ─────────────────────────────────────────────
    # SCRAPING
    # ─────────────────────────────────────────────

    def scrape_article(self, url: str) -> Optional[str]:
        """Scrape and return article text, capped at 3000 chars."""
        try:
            response = self.session.get(url, headers=self.headers, timeout=7)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) > 150:
                return text[:3000]

            return None

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout scraping: {url}")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP {e.response.status_code} scraping: {url}")
        except Exception as e:
            logger.warning(f"Scrape failed for {url}: {e}")

        return None

    # ─────────────────────────────────────────────
    # EVIDENCE FETCHING
    # ─────────────────────────────────────────────

    def fetch_evidence(self, query: str) -> List[dict]:
        """
        Fetches live news evidence for a claim via Google News RSS.
        - Resolves Google News redirects to real article URLs
        - Scrapes full article content where possible
        - Falls back to headline if scraping fails
        - Sorts trusted sources first
        """
        evidence = []
        logger.info(f"🕵️  Starting evidence search for: {query}")

        try:
            formatted_query = query.replace(" ", "+")
            rss_url = (
                f"https://news.google.com/rss/search"
                f"?q={formatted_query}&hl=en-IN&gl=IN&ceid=IN:en"
            )

            response = self.session.get(rss_url, headers=self.headers, timeout=6)
            response.raise_for_status()

            # Use lxml-xml parser for proper XML handling
            soup = BeautifulSoup(response.text, "lxml-xml")
            items = soup.find_all("item")

            logger.info(f"   → Found {len(items)} RSS items. Processing...")

            scraped_count = 0

            for item in items[:12]:
                if scraped_count >= 5:
                    break

                title_tag = item.find("title")
                link_tag = item.find("link")

                title = title_tag.get_text(strip=True) if title_tag else "No Title"
                raw_url = link_tag.get_text(strip=True) if link_tag else None

                if not raw_url:
                    continue

                # Step 1: Resolve Google News redirect → real URL
                try:
                    resolved = self.session.get(
                        raw_url,
                        headers=self.headers,
                        timeout=6,
                        allow_redirects=True
                    )
                    real_url = resolved.url
                except Exception as e:
                    logger.warning(f"Redirect failed for '{title}': {e}")
                    evidence.append({
                        "url": raw_url,
                        "title": title,
                        "text": f"[Headline only — redirect failed]: {title}",
                        "trusted": False
                    })
                    continue

                trusted = self.is_trusted(real_url)
                trust_label = "✅ TRUSTED" if trusted else "🔵 Unknown"
                logger.info(f"   {trust_label} | {real_url[:90]}")

                # Step 2: Scrape full article
                content = self.scrape_article(real_url)

                if content:
                    logger.info(f"   ✔ Scraped {len(content)} chars.")
                    evidence.append({
                        "url": real_url,
                        "title": title,
                        "text": content,
                        "trusted": trusted
                    })
                    scraped_count += 1
                else:
                    logger.info(f"   ⚠ Scrape failed — using headline fallback.")
                    evidence.append({
                        "url": real_url,
                        "title": title,
                        "text": f"[Headline only — scrape failed]: {title}",
                        "trusted": trusted
                    })

            # Trusted sources first so AI sees them prominently
            evidence.sort(key=lambda x: x["trusted"], reverse=True)

            logger.info(
                f"\n   ✅ Evidence ready: {len(evidence)} sources "
                f"({scraped_count} fully scraped, "
                f"{sum(1 for e in evidence if e['trusted'])} trusted)\n"
            )

        except Exception as e:
            logger.error(f"🔴 Evidence fetch error: {e}")

        return evidence

    # ─────────────────────────────────────────────
    # CLAIM VERIFICATION
    # ─────────────────────────────────────────────

    def verify_claim(self, claim: str) -> dict:
        # ── Input validation ─────────────────────────────────────────────────
        error = self.validate_claim(claim)
        if error:
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "explanation": error,
                "sources": [],
                "trusted_sources": []
            }

        evidence_data = self.fetch_evidence(claim)

        # ── Short-circuit if no evidence found ───────────────────────────────
        # Still call the AI — it can use its internal knowledge — but warn it.
        if not evidence_data:
            logger.warning("No evidence found; proceeding with AI internal knowledge only.")

        # ── Build trust-aware evidence summary for the prompt ────────────────
        evidence_summary = []
        for e in evidence_data:
            trust_tag = "[TRUSTED SOURCE]" if e["trusted"] else "[UNKNOWN SOURCE]"
            evidence_summary.append(
                f"{trust_tag} | {e['title']}\nURL: {e['url']}\nContent: {e['text']}"
            )

        evidence_block = (
            "\n\n---\n\n".join(evidence_summary)
            if evidence_summary
            else "No live evidence could be retrieved. Use your internal knowledge only."
        )

        system_prompt = """You are an elite, highly skeptical fact-checking AI with deep knowledge of world geography, history, science, and current events. Your job is to rigorously verify claims using the provided live evidence AND your internal knowledge.

CRITICAL REASONING RULES:

1. ABSOLUTE FACT OVERRIDE
   If a claim violates a fundamental fact — wrong country for a city, wrong Chief Minister, flat earth, etc. — mark it FAKE instantly with near-100% confidence. No news headline can override an established world fact.

2. KEYWORD TRAP RULE
   Words from the claim appearing in headlines does NOT make the claim true. If the news context is unrelated to the claim, treat that evidence as irrelevant and mark FAKE.

3. TRUSTED SOURCE WEIGHTING
   Evidence tagged [TRUSTED SOURCE] (fact-checkers, major news agencies) carries significantly more weight than [UNKNOWN SOURCE]. If a trusted source explicitly debunks or confirms a claim, weight it heavily.

4. MAJOR EVENT BURDEN OF PROOF
   For significant claims (elections, deaths, wars, disasters), evidence must overwhelmingly and directly confirm it. Weak or tangential evidence → FAKE or MISLEADING.

5. MISLEADING vs FAKE
   Use MISLEADING when the core fact is real but the framing, context, or implication distorts it. Use FAKE only when the claim is factually false.

6. UNCLEAR THRESHOLD
   Reserve UNCLEAR only for genuinely unverifiable personal or niche claims with zero relevant evidence. Do not use it to avoid making a call.

7. EXPLAIN YOUR REASONING
   In your explanation: (a) cite which sources you used and why, (b) explain what evidence you rejected and why, (c) state the true facts if the claim is wrong.

Output ONLY valid JSON with these exact keys:
- "verdict": one of REAL, FAKE, MISLEADING, or UNCLEAR
- "confidence": integer 0-100
- "explanation": detailed reasoning string"""

        user_prompt = (
            f"CLAIM TO VERIFY: {claim}\n\n"
            f"LIVE EVIDENCE ({len(evidence_data)} sources):\n\n"
            + evidence_block
        )

        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1024
            )

            raw = completion.choices[0].message.content
            result = json.loads(raw)

            for key in ("verdict", "confidence", "explanation"):
                if key not in result:
                    raise ValueError(f"Missing key in AI response: {key}")

            result["sources"] = [e["url"] for e in evidence_data]
            result["trusted_sources"] = [e["url"] for e in evidence_data if e["trusted"]]
            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "explanation": "AI returned malformed JSON. Please try again.",
                "sources": [],
                "trusted_sources": []
            }
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "explanation": str(e),
                "sources": [],
                "trusted_sources": []
            }