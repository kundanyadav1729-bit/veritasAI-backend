import os
import re
import json
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Injected into every prompt so the LLM is temporally anchored
CURRENT_DATE: str = datetime.now(timezone.utc).strftime("%B %d, %Y")


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
            "factcheck.org", "theprint.in", "scroll.in", "thewire.in",
        ]

        # Full UA string — truncated UA can trigger bot-detection on some sites
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

    # ─────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────

    def is_trusted(self, url: str) -> bool:
        return any(src in url.lower() for src in self.trusted_sources)

    def validate_claim(self, claim: str) -> Optional[str]:
        """Returns an error string if the claim is invalid, else None."""
        stripped = claim.strip() if claim else ""
        if not stripped:
            return "Claim cannot be empty."
        if len(stripped) < 10:
            return "Claim is too short to fact-check meaningfully."
        if len(stripped) > 1000:
            return "Claim is too long. Please shorten it to under 1000 characters."
        return None

    def _parse_pub_date(self, item) -> Optional[str]:
        """
        Extract a normalized publication date from an RSS <item>.
        Returns a human-readable string like '2024-11-15 08:30 UTC', or None.
        """
        tag = item.find("pubDate")
        if not tag:
            return None
        raw = tag.get_text(strip=True)
        try:
            dt = parsedate_to_datetime(raw)
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            # Return raw text if RFC-2822 parsing fails
            return raw if raw else None

    def _classify_evidence_quality(self, evidence_data: List[dict]) -> str:
        """
        Assign a quality tier used to calibrate the LLM's confidence ceiling.

        Tiers (worst → best):
          NO_EVIDENCE     – nothing fetched at all
          HEADLINES_ONLY  – items exist but none have full text
          LOW_TRUST       – full text available but no trusted-source coverage
          MEDIUM_QUALITY  – partial trusted or partial full-text coverage
          HIGH_QUALITY    – ≥2 trusted sources with full text
        """
        total = len(evidence_data)
        if total == 0:
            return "NO_EVIDENCE"

        fully_scraped = sum(
            1 for e in evidence_data if e.get("quality") == "full_text"
        )
        trusted_count = sum(1 for e in evidence_data if e["trusted"])

        if fully_scraped == 0:
            return "HEADLINES_ONLY"
        if trusted_count == 0:
            return "LOW_TRUST"
        if trusted_count >= 2 and fully_scraped >= 2:
            return "HIGH_QUALITY"
        return "MEDIUM_QUALITY"

    # ─────────────────────────────────────────────
    # SCRAPING
    # ─────────────────────────────────────────────

    def scrape_article(self, url: str) -> Optional[str]:
        """Scrape and return paragraph text from an article URL, capped at 3 000 chars."""
        try:
            response = self.session.get(url, headers=self.headers, timeout=7)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)
            text = re.sub(r"\s+", " ", text).strip()

            return text[:3000] if len(text) > 150 else None

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
        Fetches live news evidence via Google News RSS.

        Each evidence dict contains:
          url, title, text, trusted (bool), pub_date (str|None), quality (str)

        quality values: "full_text" | "headline_only"
        """
        evidence: List[dict] = []
        logger.info(f"🕵️  Starting evidence search for: {query}")

        try:
            formatted_query = query.replace(" ", "+")
            rss_url = (
                f"https://news.google.com/rss/search"
                f"?q={formatted_query}&hl=en-IN&gl=IN&ceid=IN:en"
            )

            response = self.session.get(rss_url, headers=self.headers, timeout=6)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml-xml")
            items = soup.find_all("item")
            logger.info(f"   → Found {len(items)} RSS items. Processing...")

            scraped_count = 0

            for item in items[:12]:
                if scraped_count >= 5:
                    break

                title_tag = item.find("title")
                link_tag = item.find("link")
                pub_date = self._parse_pub_date(item)

                title = title_tag.get_text(strip=True) if title_tag else "No Title"

                # ── lxml-xml quirk: <link> is often a NavigableString sibling ──
                raw_url: Optional[str] = None
                if link_tag:
                    raw_url = link_tag.get_text(strip=True) or None
                if not raw_url and link_tag:
                    sib = link_tag.next_sibling
                    if sib and isinstance(sib, str):
                        raw_url = sib.strip() or None

                if not raw_url:
                    logger.warning(f"   ⚠ Could not extract URL for: {title}")
                    continue

                # ── Step 1: Resolve Google News redirect → real article URL ──
                try:
                    resolved = self.session.get(
                        raw_url,
                        headers=self.headers,
                        timeout=6,
                        allow_redirects=True,
                    )
                    real_url = resolved.url
                except Exception as e:
                    logger.warning(f"Redirect failed for '{title}': {e}")
                    evidence.append({
                        "url": raw_url,
                        "title": title,
                        "text": f"[Headline only — redirect failed]: {title}",
                        "trusted": False,
                        "pub_date": pub_date,
                        "quality": "headline_only",
                    })
                    continue

                trusted = self.is_trusted(real_url)
                logger.info(
                    f"   {'✅ TRUSTED' if trusted else '🔵 Unknown'} | {real_url[:90]}"
                )

                # ── Step 2: Scrape full article ──
                content = self.scrape_article(real_url)

                if content:
                    logger.info(f"   ✔ Scraped {len(content)} chars.")
                    evidence.append({
                        "url": real_url,
                        "title": title,
                        "text": content,
                        "trusted": trusted,
                        "pub_date": pub_date,
                        "quality": "full_text",
                    })
                    scraped_count += 1
                else:
                    logger.info("   ⚠ Scrape failed — using headline fallback.")
                    evidence.append({
                        "url": real_url,
                        "title": title,
                        "text": f"[Headline only — scrape failed]: {title}",
                        "trusted": trusted,
                        "pub_date": pub_date,
                        "quality": "headline_only",
                    })

            # Trusted sources first — LLM processes them with higher attention
            evidence.sort(key=lambda x: (x["trusted"], x["quality"] == "full_text"), reverse=True)

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
        # ── Input validation ──────────────────────────────────────────────────
        error = self.validate_claim(claim)
        if error:
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "explanation": error,
                "correction": None,
                "knowledge_cutoff_warning": False,
                "evidence_quality": "NONE",
                "sources": [],
                "trusted_sources": [],
            }

        evidence_data = self.fetch_evidence(claim)
        evidence_quality = self._classify_evidence_quality(evidence_data)

        if not evidence_data:
            logger.warning("No evidence found — proceeding with AI internal knowledge only.")

        # ── Build evidence block for the prompt ──────────────────────────────
        evidence_lines: List[str] = []
        for e in evidence_data:
            trust_tag  = "[TRUSTED SOURCE]" if e["trusted"] else "[UNKNOWN SOURCE]"
            text_tag   = "[FULL TEXT]"       if e.get("quality") == "full_text" else "[HEADLINE ONLY]"
            date_tag   = f"[Published: {e['pub_date']}]" if e.get("pub_date") else "[Published: Unknown]"

            evidence_lines.append(
                f"{trust_tag} {text_tag} {date_tag}\n"
                f"Title: {e['title']}\n"
                f"URL: {e['url']}\n"
                f"Content: {e['text']}"
            )

        evidence_block = (
            "\n\n---\n\n".join(evidence_lines)
            if evidence_lines
            else "No live evidence was retrieved."
        )

        # ── Per-quality advisory injected into the prompt ────────────────────
        quality_advisories = {
            "NO_EVIDENCE": (
                "⚠️  CRITICAL — No live evidence was retrieved. "
                "You are operating on internal training knowledge alone. "
                "For ANY time-sensitive claim (politics, appointments, deaths, sports results, "
                "recent events), you MUST set confidence ≤ 40 and "
                "knowledge_cutoff_warning = true. "
                "Do NOT present stale training-data facts as current reality."
            ),
            "HEADLINES_ONLY": (
                "⚠️  WARNING — All evidence is headline-only; no full article text was fetched. "
                "Headlines are often incomplete or misleading without body context. "
                "Reduce your natural confidence estimate by ~20 points. "
                "Do not treat a keyword match in a headline as confirmation of the claim."
            ),
            "LOW_TRUST": (
                "⚠️  NOTE — No verified/trusted sources are present in the evidence set. "
                "Treat all retrieved evidence as provisional. "
                "For well-known stable facts, your internal knowledge may be more reliable "
                "than these unverified sources."
            ),
            "MEDIUM_QUALITY": (
                "ℹ️  Evidence quality is moderate. "
                "Weight [TRUSTED SOURCE] items heavily over [UNKNOWN SOURCE] items."
            ),
            "HIGH_QUALITY": (
                "✅ Evidence quality is high. "
                "Multiple trusted sources with full text are available — use them as the primary basis."
            ),
        }
        quality_advisory = quality_advisories.get(evidence_quality, "")

        # ── System prompt ─────────────────────────────────────────────────────
        system_prompt = f"""You are a rigorous, neutral fact-checking analyst. Today's date is {CURRENT_DATE}.

Verify the given claim using the live evidence provided AND your internal knowledge.
Follow ALL rules below without exception.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — KNOWLEDGE CUTOFF DISCIPLINE  (most important)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your training data has a cutoff; you cannot know current events with certainty.

• Time-sensitive claims (elections, appointments, deaths, disasters, sports, laws):
  – Live evidence CONFIRMS it → cite it, use it, set knowledge_cutoff_warning = false.
  – Live evidence is absent or weak → set confidence ≤ 40, set knowledge_cutoff_warning = true.
  – NEVER assert a time-sensitive fact as certain when relying solely on training data.

• Timeless facts (geography, settled science, pre-2023 history) → internal knowledge is reliable.

• Always write "Based on internal training knowledge (not verified by live evidence):" when
  you are using only your training data for a factual assertion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — ANTI-SENSATIONALISM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your explanation must be calm, neutral, and factual.

FORBIDDEN words/phrases in your explanation:
  shocking, explosive, bombshell, alarming, stunning, jaw-dropping, outrageous,
  devastating, earth-shattering, breaks the internet, goes viral.

Do NOT:
  • Use emotionally charged or hyperbolic language.
  • Assign political blame beyond what the evidence directly supports.
  • Exaggerate certainty to make the result sound more dramatic.

Write as a neutral analyst, not a journalist optimising for clicks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 — ABSOLUTE FACT OVERRIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If a claim contradicts an established, timeless fact (wrong country for a city,
basic science denial, wrong capital, etc.) → FAKE, confidence ≥ 90.
No headline can override a well-established world fact.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 4 — KEYWORD TRAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Keywords from the claim appearing in a headline do NOT confirm the claim.
If the article's actual context is unrelated to what the claim asserts,
discard that evidence as irrelevant and note why.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 5 — SOURCE WEIGHTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[TRUSTED SOURCE] carries far more weight than [UNKNOWN SOURCE].
[HEADLINE ONLY] carries far less weight than [FULL TEXT].
[Published: Unknown] or old dates reduce recency weight.
If a trusted source explicitly debunks the claim → weight heavily toward FAKE/MISLEADING.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 6 — VERDICT DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL        – claim is factually accurate and the evidence directly supports it.
FAKE        – claim is factually false.
MISLEADING  – core fact is real, but framing, context, or implication distorts it.
UNCLEAR     – genuinely unverifiable; use sparingly only when zero relevant evidence exists.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 7 — EXPLANATION STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your explanation MUST cover:
  (a) Which sources you used and why.
  (b) Which evidence you rejected and why.
  (c) The correct facts if the claim is wrong.
  (d) An explicit flag if you used internal training knowledge instead of live evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE QUALITY ADVISORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{quality_advisory}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY valid JSON with these exact keys — no preamble, no markdown fences:

{{
  "verdict":                  "REAL" | "FAKE" | "MISLEADING" | "UNCLEAR",
  "confidence":               <integer 0–100>,
  "explanation":              "<neutral, structured reasoning string>",
  "correction":               "<the correct fact if FAKE or MISLEADING, else null>",
  "knowledge_cutoff_warning": <true | false>
}}"""

        user_prompt = (
            f"CLAIM TO VERIFY: {claim}\n\n"
            f"TODAY'S DATE: {CURRENT_DATE}\n\n"
            f"EVIDENCE QUALITY TIER: {evidence_quality}\n\n"
            f"LIVE EVIDENCE ({len(evidence_data)} sources):\n\n"
            + evidence_block
        )

        # ── LLM call ──────────────────────────────────────────────────────────
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1200,
            )

            raw = completion.choices[0].message.content
            result = json.loads(raw)

            # ── Validate required keys ────────────────────────────────────────
            for key in ("verdict", "confidence", "explanation"):
                if key not in result:
                    raise ValueError(f"Missing required key in AI response: '{key}'")

            # ── Normalise and clamp fields ────────────────────────────────────
            result["verdict"] = result["verdict"].upper().strip()
            if result["verdict"] not in ("REAL", "FAKE", "MISLEADING", "UNCLEAR"):
                logger.warning(
                    f"Unexpected verdict value '{result['verdict']}' — defaulting to UNCLEAR"
                )
                result["verdict"] = "UNCLEAR"

            result["confidence"] = max(0, min(100, int(result.get("confidence", 0))))
            result["knowledge_cutoff_warning"] = bool(
                result.get("knowledge_cutoff_warning", False)
            )
            result["correction"] = result.get("correction") or None

            # ── Attach source metadata ────────────────────────────────────────
            result["evidence_quality"]  = evidence_quality
            result["sources"]           = [e["url"] for e in evidence_data]
            result["trusted_sources"]   = [e["url"] for e in evidence_data if e["trusted"]]

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "explanation": "The AI returned malformed JSON. Please try again.",
                "correction": None,
                "knowledge_cutoff_warning": False,
                "evidence_quality": evidence_quality,
                "sources": [],
                "trusted_sources": [],
            }
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "explanation": str(e),
                "correction": None,
                "knowledge_cutoff_warning": False,
                "evidence_quality": evidence_quality,
                "sources": [],
                "trusted_sources": [],
            }