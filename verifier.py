
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

CURRENT_DATE: str = datetime.now(timezone.utc).strftime("%B %d, %Y")

# Wikipedia blocks generic Chrome UAs — must use a descriptive bot UA
WIKI_HEADERS = {
    "User-Agent": "FakeNewsDetector/1.0 (educational project; python-requests) python-requests/2.31"
}


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
            "en.wikipedia.org",
        ]

        # Full Chrome UA for news sites
        self.news_headers = {
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
        stripped = claim.strip() if claim else ""
        if not stripped:
            return "Claim cannot be empty."
        if len(stripped) < 10:
            return "Claim is too short to fact-check meaningfully."
        if len(stripped) > 1000:
            return "Claim is too long. Please shorten it to under 1000 characters."
        return None

    def _parse_pub_date(self, item) -> Optional[str]:
        tag = item.find("pubDate")
        if not tag:
            return None
        raw = tag.get_text(strip=True)
        try:
            dt = parsedate_to_datetime(raw)
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return raw if raw else None

    def _classify_evidence_quality(self, evidence_data: List[dict]) -> str:
        if not evidence_data:
            return "NO_EVIDENCE"
        fully_scraped = sum(1 for e in evidence_data if e.get("quality") == "full_text")
        trusted_count = sum(1 for e in evidence_data if e["trusted"])
        if fully_scraped == 0:
            return "HEADLINES_ONLY"
        if trusted_count == 0:
            return "LOW_TRUST"
        if trusted_count >= 2 and fully_scraped >= 2:
            return "HIGH_QUALITY"
        return "MEDIUM_QUALITY"

    def _extract_subject(self, claim: str) -> str:
        """Extract person/subject name from a claim string."""
        match = re.match(
            r"^([A-Z][a-zA-Z\s\.\-]{2,40?})\s+(?:is|are|was|were|became|has|have|had)\b",
            claim.strip()
        )
        if match:
            return match.group(1).strip()
        return claim.strip()

    def _extract_political_position(self, claim: str) -> Optional[str]:
        """
        Detect and extract a political position from a claim.
        Returns a Wikipedia-searchable position string, or None.

        Examples:
          'Suvendu Adhikari is the CM of West Bengal'
              → 'Chief Minister of West Bengal'
          'Modi is the PM of India'
              → 'Prime Minister of India'
        """
        claim_lower = claim.lower()

        # Map of abbreviations / keywords → full Wikipedia title fragment
        position_patterns = [
            (r'\b(chief minister|cm)\b.*?\bof\s+([a-z\s]+)', 'Chief Minister of {}'),
            (r'\b(prime minister|pm)\b.*?\bof\s+([a-z\s]+)', 'Prime Minister of {}'),
            (r'\b(president)\b.*?\bof\s+([a-z\s]+)', 'President of {}'),
            (r'\b(governor)\b.*?\bof\s+([a-z\s]+)', 'Governor of {}'),
            (r'\b(deputy chief minister|deputy cm)\b.*?\bof\s+([a-z\s]+)', 'Deputy Chief Minister of {}'),
        ]

        for pattern, template in position_patterns:
            m = re.search(pattern, claim_lower)
            if m:
                # Last captured group is the place name
                place = m.group(m.lastindex).strip().title()
                # Clean trailing noise words
                place = re.sub(r'\b(the|a|an|its|his|her)\b', '', place).strip()
                return template.format(place)

        return None

    # ─────────────────────────────────────────────
    # SCRAPING
    # ─────────────────────────────────────────────

    def scrape_article(self, url: str) -> Optional[str]:
        """Scrape paragraph text from a URL, capped at 3000 chars."""
        try:
            response = self.session.get(url, headers=self.news_headers, timeout=7)
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
    # WIKIPEDIA FETCH
    # ─────────────────────────────────────────────

    def _wiki_fetch(self, title: str) -> Optional[dict]:
        """
        Internal helper — fetch one Wikipedia article by title.
        Uses the MediaWiki action API with a bot-friendly User-Agent.
        Returns an evidence dict or None.
        """
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
            "redirects": 1,
        }
        try:
            r = self.session.get(url, params=params, headers=WIKI_HEADERS, timeout=8)
            if r.status_code != 200:
                logger.warning(f"   📖 Wikipedia HTTP {r.status_code} for '{title}'")
                return None

            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                if pid == "-1":
                    logger.info(f"   📖 Wikipedia: no article found for '{title}'")
                    return None
                extract = page.get("extract", "").strip()
                if len(extract) > 150:
                    page_title = page.get("title", title)
                    page_url = (
                        f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                    )
                    logger.info(f"   📖 Wikipedia hit: '{page_title}'")
                    return {
                        "url": page_url,
                        "title": f"Wikipedia — {page_title}",
                        "text": extract[:3000],
                        "trusted": True,
                        "pub_date": None,
                        "quality": "full_text",
                        "source_type": "wikipedia",
                    }
        except Exception as e:
            logger.warning(f"   📖 Wikipedia fetch failed for '{title}': {e}")
        return None

    def fetch_wikipedia_evidence(self, claim: str) -> List[dict]:
        """
        Fetch up to TWO Wikipedia articles per claim:

          1. The POSITION page  e.g. 'Chief Minister of West Bengal'
             → Lists who currently holds that office. Most up-to-date.

          2. The PERSON page    e.g. 'Suvendu Adhikari'
             → May lag behind breaking changes but gives biographical context.

        Fetching both lets the LLM cross-reference them and detect staleness.
        """
        results: List[dict] = []

        # -- Article 1: position page (most reliable for 'who holds X office') --
        position = self._extract_political_position(claim)
        if position:
            logger.info(f"   📖 Fetching Wikipedia position page: '{position}'")
            wiki = self._wiki_fetch(position)
            if wiki:
                wiki["wiki_type"] = "position_page"
                results.append(wiki)

        # -- Article 2: person/subject page --
        subject = self._extract_subject(claim)
        if subject and subject.lower() != claim.strip().lower():
            logger.info(f"   📖 Fetching Wikipedia person page: '{subject}'")
            wiki2 = self._wiki_fetch(subject)
            if wiki2:
                wiki2["wiki_type"] = "person_page"
                results.append(wiki2)

        return results

    # ─────────────────────────────────────────────
    # EVIDENCE FETCHING
    # ─────────────────────────────────────────────

    def fetch_evidence(self, query: str) -> List[dict]:
        """
        Fetches live evidence from two sources:
          1. Google News RSS  — recent news (headlines + scraped text where possible)
          2. Wikipedia        — position page + person page for political claims

        Each evidence dict:
          url, title, text, trusted (bool), pub_date (str|None),
          quality ('full_text'|'headline_only'), source_type (str)
        """
        evidence: List[dict] = []
        logger.info(f"🕵️  Starting evidence search for: {query}")

       # ── Source 1: Google News RSS ─────────────────────────────────────────
        try:
            # Strip noise words so RSS actually finds articles
            noise_words = {"is", "are", "the", "new", "of", "a", "an", "that", "was", "were", "has", "have", "been"}
            clean_query_words = [w for w in query.split() if w.lower() not in noise_words]
            formatted_query = "+".join(clean_query_words)
            
            rss_url = (
                f"https://news.google.com/rss/search"
                f"?q={formatted_query}&hl=en-IN&gl=IN&ceid=IN:en"
            )
            response = self.session.get(rss_url, headers=self.news_headers, timeout=6)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml-xml")
            items = soup.find_all("item")
            logger.info(f"   → Found {len(items)} RSS items. Processing...")

            scraped_count = 0
            for item in items[:12]:
                if scraped_count >= 5:
                    break

                title_tag = item.find("title")
                link_tag  = item.find("link")
                pub_date  = self._parse_pub_date(item)
                title     = title_tag.get_text(strip=True) if title_tag else "No Title"

                # lxml-xml quirk: URL may be a sibling text node not tag content
                raw_url: Optional[str] = None
                if link_tag:
                    raw_url = link_tag.get_text(strip=True) or None
                if not raw_url and link_tag:
                    sib = link_tag.next_sibling
                    if sib and isinstance(sib, str):
                        raw_url = sib.strip() or None
                if not raw_url:
                    logger.warning(f"   ⚠ No URL for: {title}")
                    continue

                # Resolve Google redirect → real article URL
                try:
                    resolved = self.session.get(
                        raw_url, headers=self.news_headers, timeout=6, allow_redirects=True
                    )
                    real_url = resolved.url
                except Exception as e:
                    logger.warning(f"Redirect failed for '{title}': {e}")
                    evidence.append({
                        "url": raw_url, "title": title,
                        "text": f"[Headline only — redirect failed]: {title}",
                        "trusted": False, "pub_date": pub_date,
                        "quality": "headline_only", "source_type": "news_rss",
                    })
                    continue

                trusted = self.is_trusted(real_url)
                logger.info(f"   {'✅ TRUSTED' if trusted else '🔵 Unknown'} | {real_url[:90]}")

                content = self.scrape_article(real_url)
                if content:
                    logger.info(f"   ✔ Scraped {len(content)} chars.")
                    evidence.append({
                        "url": real_url, "title": title, "text": content,
                        "trusted": trusted, "pub_date": pub_date,
                        "quality": "full_text", "source_type": "news_rss",
                    })
                    scraped_count += 1
                else:
                    logger.info("   ⚠ Scrape failed — headline fallback.")
                    evidence.append({
                        "url": real_url, "title": title,
                        "text": f"[Headline only — scrape failed]: {title}",
                        "trusted": trusted, "pub_date": pub_date,
                        "quality": "headline_only", "source_type": "news_rss",
                    })

        except Exception as e:
            logger.error(f"🔴 Google News RSS error: {e}")

        # ── Source 2: Wikipedia (position page + person page) ─────────────────
        wiki_results = self.fetch_wikipedia_evidence(query)
        evidence.extend(wiki_results)

        if not wiki_results:
            logger.info("   📖 No usable Wikipedia articles found for this claim.")

        # Sort: trusted + full_text first
        evidence.sort(
            key=lambda x: (x["trusted"], x["quality"] == "full_text"),
            reverse=True
        )

        logger.info(
            f"\n   ✅ Evidence ready: {len(evidence)} sources "
            f"({sum(1 for e in evidence if e.get('quality') == 'full_text')} full text, "
            f"{sum(1 for e in evidence if e['trusted'])} trusted)\n"
        )
        return evidence

    # ─────────────────────────────────────────────
    # CLAIM VERIFICATION
    # ─────────────────────────────────────────────

    def verify_claim(self, claim: str) -> dict:

        # ── Input validation ──────────────────────────────────────────────────
        error = self.validate_claim(claim)
        if error:
            return {
                "verdict": "ERROR", "confidence": 0, "explanation": error,
                "correction": None, "knowledge_cutoff_warning": False,
                "evidence_quality": "NONE", "sources": [], "trusted_sources": [],
            }

        evidence_data = self.fetch_evidence(claim)
        evidence_quality = self._classify_evidence_quality(evidence_data)

        if not evidence_data:
            logger.warning("No evidence found — AI internal knowledge only.")

        # ── Build evidence block ──────────────────────────────────────────────
        evidence_lines: List[str] = []
        for e in evidence_data:
            trust_tag  = "[TRUSTED SOURCE]" if e["trusted"] else "[UNKNOWN SOURCE]"
            text_tag   = "[FULL TEXT]" if e.get("quality") == "full_text" else "[HEADLINE ONLY]"
            date_tag   = f"[Published: {e['pub_date']}]" if e.get("pub_date") else "[Published: Unknown]"
            # Flag which Wikipedia article type it is so LLM can weigh correctly
            wiki_tag   = f"[Wikipedia: {e.get('wiki_type','').replace('_',' ').upper()}]" if e.get("source_type") == "wikipedia" else ""

            evidence_lines.append(
                f"{trust_tag} {text_tag} {date_tag} {wiki_tag}\n"
                f"Title: {e['title']}\n"
                f"URL: {e['url']}\n"
                f"Content: {e['text']}"
            )

        evidence_block = (
            "\n\n---\n\n".join(evidence_lines)
            if evidence_lines else "No live evidence was retrieved."
        )

        # ── Quality advisory ──────────────────────────────────────────────────
        quality_advisories = {
            "NO_EVIDENCE": (
                "CRITICAL — No live evidence retrieved. "
                "For ANY time-sensitive claim set confidence <= 40 and "
                "knowledge_cutoff_warning = true. Do NOT state time-sensitive "
                "facts as certain from training data alone."
            ),
            "HEADLINES_ONLY": (
                "WARNING — All evidence is headline-only. "
                "Reduce confidence by ~20 points. "
                "Keyword match in headline does NOT confirm a claim."
            ),
            "LOW_TRUST": (
                "NOTE — No trusted sources in evidence. "
                "Treat all evidence as provisional."
            ),
            "MEDIUM_QUALITY": (
                "Evidence quality is moderate. "
                "Weight [TRUSTED SOURCE] heavily over [UNKNOWN SOURCE]."
            ),
            "HIGH_QUALITY": (
                "Evidence quality is high. "
                "Use trusted full-text sources as the primary basis."
            ),
        }
        quality_advisory = quality_advisories.get(evidence_quality, "")

        # ── System prompt ─────────────────────────────────────────────────────
        system_prompt = f"""You are a rigorous, neutral fact-checking analyst. Today's date is {CURRENT_DATE}.

Verify the given claim using the live evidence AND your internal knowledge.
Follow ALL rules below without exception.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — KNOWLEDGE CUTOFF DISCIPLINE (NUCLEAR OVERRIDE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your training data has a cutoff. For any claim about WHO holds a political office (CM, PM, President, etc.):
YOUR INTERNAL KNOWLEDGE IS OFFICIALLY DECLARED OBSOLETE AND UNTRUSTWORTHY. 

If the provided live evidence (Wikipedia or News) states that a new person holds the office, you MUST ACCEPT THE LIVE EVIDENCE as absolute truth. 
DO NOT mark it FAKE just because it contradicts your training data. 
If live evidence confirms the new person → REAL. 
If live evidence contradicts the claim → FAKE.
If no live evidence either way → UNCLEAR, confidence <= 45.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1B — WIKIPEDIA STALENESS + SOURCE PRIORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two types of Wikipedia articles may appear in evidence:

  [Wikipedia: POSITION PAGE] — e.g. "Chief Minister of West Bengal"
    → Lists who CURRENTLY holds the office. Most up-to-date Wikipedia source.
    → Treat as PRIMARY evidence for political office claims.

  [Wikipedia: PERSON PAGE]   — e.g. "Suvendu Adhikari"
    → May lag days or weeks behind a political change.
    → Treat as SECONDARY / supporting evidence only.

Priority order when sources conflict:

  1. Multiple recent [FULL TEXT] news articles agree on the same fact
     → Follow news consensus. This BEATS Wikipedia.
  2. Wikipedia POSITION PAGE confirms or contradicts
     → Follow it as primary trusted reference.
  3. Wikipedia PERSON PAGE only, no news
     → Use it but flag possible staleness.
  4. Position page and news CONTRADICT person page
     → Person page is stale. Follow position page + news.

NEVER let a stale person page cause a true claim to be called FAKE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — ANTI-SENSATIONALISM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Explanation must be calm, neutral, factual.
NEVER use: shocking, explosive, bombshell, alarming, stunning,
jaw-dropping, outrageous, devastating, earth-shattering.
Do not assign political blame beyond what evidence directly supports.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 — ABSOLUTE FACT OVERRIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If a claim contradicts a timeless established fact (wrong country for a city,
basic science denial, etc.) → FAKE, confidence >= 90.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 4 — KEYWORD TRAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Keywords from the claim in a headline do NOT confirm the claim.
If article context is unrelated to the claim, discard that evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 5 — SOURCE WEIGHTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. [TRUSTED] + [FULL TEXT] + recent date  ← highest weight
  2. [TRUSTED] + [FULL TEXT] + unknown date ← high weight
  3. [TRUSTED] + [HEADLINE ONLY]            ← medium weight
  4. [UNKNOWN] + [FULL TEXT]                ← low weight
  5. [UNKNOWN] + [HEADLINE ONLY]            ← lowest weight

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 6 — VERDICT DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REAL       — claim is accurate; evidence directly supports it.
FAKE       — claim is false; evidence directly contradicts it.
MISLEADING — core fact is real but framing/context distorts it.
UNCLEAR    — genuinely unverifiable; use sparingly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 7 — EXPLANATION STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Explanation must cover:
  (a) Which sources you used and why.
  (b) Which evidence you rejected and why.
  (c) Correct facts if claim is wrong.
  (d) Explicit flag when using internal knowledge instead of live evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE QUALITY ADVISORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{quality_advisory}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — valid JSON only, no preamble, no markdown fences
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "verdict":                  "REAL" | "FAKE" | "MISLEADING" | "UNCLEAR",
  "confidence":               <integer 0-100>,
  "explanation":              "<neutral structured reasoning>",
  "correction":               "<correct fact if FAKE or MISLEADING, else null>",
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

            raw    = completion.choices[0].message.content
            result = json.loads(raw)

            for key in ("verdict", "confidence", "explanation"):
                if key not in result:
                    raise ValueError(f"Missing key: '{key}'")

            result["verdict"] = result["verdict"].upper().strip()
            if result["verdict"] not in ("REAL", "FAKE", "MISLEADING", "UNCLEAR"):
                logger.warning(f"Unexpected verdict '{result['verdict']}' → UNCLEAR")
                result["verdict"] = "UNCLEAR"

            result["confidence"]             = max(0, min(100, int(result.get("confidence", 0))))
            result["knowledge_cutoff_warning"] = bool(result.get("knowledge_cutoff_warning", False))
            result["correction"]             = result.get("correction") or None
            result["evidence_quality"]       = evidence_quality
            result["sources"]                = [e["url"] for e in evidence_data]
            result["trusted_sources"]        = [e["url"] for e in evidence_data if e["trusted"]]

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {
                "verdict": "ERROR", "confidence": 0,
                "explanation": "AI returned malformed JSON. Please try again.",
                "correction": None, "knowledge_cutoff_warning": False,
                "evidence_quality": evidence_quality, "sources": [], "trusted_sources": [],
            }
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {
                "verdict": "ERROR", "confidence": 0, "explanation": str(e),
                "correction": None, "knowledge_cutoff_warning": False,
                "evidence_quality": evidence_quality, "sources": [], "trusted_sources": [],
            }

