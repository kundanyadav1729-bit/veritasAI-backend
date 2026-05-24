import os
import re
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class Verifier:
    def __init__(self):
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.session = requests.Session()
        
        # VIP Whitelist - Indian & Global Trusted Sources
        self.trusted_sources = [
            "thehindu.com", "ndtv.com", "indiatoday.in", "indianexpress.com", 
            "hindustantimes.com", "timesofindia.indiatimes.com", "news18.com",
            "altnews.in", "boomlive.in", "pib.gov.in", "vishvasnews.com",
            "reuters.com", "apnews.com", "bbc.com", "snopes.com", "politifact.com"
        ]

    def is_trusted(self, url):
        return any(trusted in url.lower() for trusted in self.trusted_sources)

    def scrape_article(self, url):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = self.session.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove scripts and styles
            for tag in soup(["script", "style"]):
                tag.decompose()
                
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text(strip=True) for p in paragraphs])
            
            if len(text) > 100:
                return text[:2500] # Cap at 2500 chars to save AI context
            return None
        except Exception:
            return None

    def fetch_evidence(self, query):
        import re
        evidence = []
        print(f"\n--- 🕵️ STARTING HEADLINE SEARCH FOR: {query} ---")
        
        try:
            formatted_query = query.replace(" ", "+")
            rss_url = f"https://news.google.com/rss/search?q={formatted_query}&hl=en-IN&gl=IN&ceid=IN:en"
            
            headers = {"User-Agent": "Mozilla/5.0"}
            response = self.session.get(rss_url, headers=headers, timeout=5)
            
            # Use regex to grab titles and links directly to avoid parser crashes
            titles = re.findall(r'<title>(.*?)</title>', response.text)
            links = re.findall(r'<link>(https://news\.google\.com/rss/articles/[^<]+)</link>', response.text)
            
            # The first title in the RSS feed is the page title, so we skip it
            news_titles = titles[1:] if len(titles) > 1 else []
            
            print(f"   -> Found {len(links)} news headlines.")
            
            # Feed the headlines directly to the AI
            for i in range(min(10, len(links))):
                title = news_titles[i] if i < len(news_titles) else "Breaking News"
                url = links[i]
                
                print(f"🟢 Feeding AI Headline: {title}")
                
                evidence.append({
                    "url": url,
                    "text": f"Breaking News Headline: {title}"
                })
                
        except Exception as e:
            print(f"🔴 Search Error: {e}")

        return evidence

    def verify_claim(self, claim):
        evidence_data = self.fetch_evidence(claim)
        
        system_prompt = """You are an elite, highly skeptical fact-checking AI. Your job is to verify claims using a combination of the provided live news evidence AND your internal knowledge of absolute, established world facts.

CRITICAL RULES:
1. THE ABSOLUTE FACT OVERRIDE: If a claim violates fundamental geographical, scientific, historical, or political facts (e.g., claiming a city is in the wrong country, the earth is flat, or the wrong person is a Chief Minister), you MUST instantly mark it FAKE with near 100% confidence. Explain the true facts. Do not let irrelevant news articles confuse you.
2. THE KEYWORD TRAP: Just because words from the claim appear in the news (like "Kolkata" and "Bangladesh" in an earthquake article) does NOT make the claim true. If the context of the news is completely different from the claim, mark it FAKE.
3. THE MAJOR EVENT RULE: For massive world events, evidence MUST overwhelmingly confirm it. If evidence is missing or irrelevant, mark it FAKE.
4. UNPROVABLE CLAIMS: Only mark UNCLEAR for random, unprovable personal statements (e.g., "My friend bought a car").

Output your response strictly in JSON format with exactly these keys: "verdict" (REAL, FAKE, MISLEADING, or UNCLEAR), "confidence" (number between 0-100), and "explanation"."""
        
        user_prompt = f"Claim: {claim}\nEvidence: {json.dumps(evidence_data)}"
        
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(completion.choices[0].message.content)
            result["sources"] = [e["url"] for e in evidence_data]
            return result
        except Exception as e:
            return {"verdict": "ERROR", "confidence": 0, "explanation": str(e), "sources": []}