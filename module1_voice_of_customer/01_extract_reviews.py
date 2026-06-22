"""
Smart Data Extractor - combines multiple sources automatically:

  1. ALWAYS tries Apple App Store first (if APP_STORE_ID is set)
  2. ALWAYS tries Google Maps via SerpAPI (nationwide coverage)
  3. Combines both into one reviews.csv

This gives maximum review coverage regardless of whether the brand has an app.
"""

import os, sys, csv, json, time, re
import urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BRAND_NAME, KEYWORDS, APP_STORE_ID, APP_COUNTRY,
    MAX_REVIEW_PAGES, DATA_DIR, REVIEWS_CSV,
)

SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "")
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

FIELDNAMES = [
    "review_id", "stars", "date", "title", "text",
    "source", "product", "version", "vote_count",
    "place_name", "address", "city", "state",
    "latitude", "longitude", "google_rating", "total_reviews_at_location",
]

# US states for nationwide search coverage
US_STATE_MAP = {
    "California":"CA","Texas":"TX","Florida":"FL","New York":"NY","Pennsylvania":"PA",
    "Illinois":"IL","Ohio":"OH","Georgia":"GA","North Carolina":"NC","Michigan":"MI",
    "New Jersey":"NJ","Virginia":"VA","Washington":"WA","Arizona":"AZ","Massachusetts":"MA",
    "Tennessee":"TN","Indiana":"IN","Missouri":"MO","Maryland":"MD","Wisconsin":"WI",
    "Colorado":"CO","Minnesota":"MN","South Carolina":"SC","Alabama":"AL","Louisiana":"LA",
    "Nevada":"NV","Oregon":"OR","Connecticut":"CT",
}
US_STATES = list(US_STATE_MAP.keys())


def fetch_url(url, timeout=20):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def parse_relative_date(text):
    if not text:
        return datetime.now().strftime("%Y-%m-%d")
    text = text.lower().strip()
    now  = datetime.now()
    try:
        if "just now" in text or "moment" in text:
            return now.strftime("%Y-%m-%d")
        num = re.search(r'\d+', text)
        n   = int(num.group()) if num else 1
        if "year"  in text: return (now - timedelta(days=n*365)).strftime("%Y-%m-%d")
        if "month" in text: return (now - timedelta(days=n*30)).strftime("%Y-%m-%d")
        if "week"  in text: return (now - timedelta(days=n*7)).strftime("%Y-%m-%d")
        if "day"   in text: return (now - timedelta(days=n)).strftime("%Y-%m-%d")
        if "hour"  in text or "minute" in text: return now.strftime("%Y-%m-%d")
    except Exception:
        pass
    return now.strftime("%Y-%m-%d")


def parse_address(raw_address, search_term=""):
    """
    Extract city and state from address string.
    Falls back to the search term's state name if the address
    doesn't contain a clean state code (common with mall/outlet addresses).
    """
    city, state = "", ""
    if raw_address:
        parts = [p.strip() for p in raw_address.split(",") if p.strip()]
        for i, part in enumerate(parts):
            m = re.search(r'\b([A-Z]{2})\b\s*\d{0,5}$', part)
            if m and m.group(1) not in ("US", "ST", "RD", "DR", "BLVD", "AVE"):
                state = m.group(1)
                if i > 0:
                    city = parts[i-1]
                break

    # Fallback: use the state we searched for (most reliable for mall addresses)
    if not state and search_term:
        for st_name, st_code in US_STATE_MAP.items():
            if st_name.lower() in search_term.lower():
                state = st_code
                break

    return city.strip(), state.strip()


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 - Apple App Store (auto-discovers app ID if not set)
# ══════════════════════════════════════════════════════════════════════════════

def auto_find_app_id(brand_name):
    """
    Search iTunes for an app matching the brand name.
    Returns (app_id, app_title) or (None, None) if no good match found.
    Free, no API key needed.
    """
    print(f"\n🔎 Auto-searching App Store for: '{brand_name}'...")
    try:
        query = urllib.parse.quote_plus(brand_name)
        url = f"https://itunes.apple.com/search?term={query}&entity=software&country={APP_COUNTRY}&limit=10"
        data = json.loads(fetch_url(url))
        results = data.get("results", [])

        if not results:
            print(f"   No App Store apps found for '{brand_name}'.")
            return None, None

        # Find best match: app title contains the brand name, has decent rating count
        brand_lower = brand_name.lower()
        candidates = []
        for app in results:
            title = app.get("trackName", "")
            rating_count = app.get("userRatingCount", 0)
            if brand_lower in title.lower():
                candidates.append((app.get("trackId"), title, rating_count))

        if not candidates:
            # Fall back to top result even if name doesn't match exactly
            top = results[0]
            app_id = str(top.get("trackId", ""))
            title  = top.get("trackName", "")
            ratings = top.get("userRatingCount", 0)
            print(f"   No exact name match. Closest result: '{title}' ({ratings} ratings)")
            if ratings < 50:
                print(f"   Too few ratings to be useful - skipping App Store.")
                return None, None
            return app_id, title

        # Pick candidate with most ratings (most likely the real official app)
        candidates.sort(key=lambda x: -x[2])
        app_id, title, ratings = candidates[0]

        print(f"   Found: '{title}' (ID: {app_id}, {ratings} ratings)")

        if ratings < 20:
            print(f"   Too few ratings ({ratings}) to be useful - skipping App Store.")
            return None, None

        return str(app_id), title

    except Exception as ex:
        print(f"   Auto-search failed: {ex}")
        return None, None


def scrape_app_store():
    app_id = APP_STORE_ID.strip()

    if not app_id:
        app_id, found_title = auto_find_app_id(BRAND_NAME)
        if not app_id:
            print("   Skipping App Store - no usable app found.")
            return []

    print(f"\n📱 Scraping Apple App Store (ID: {app_id})...")
    reviews = []
    for page in range(1, MAX_REVIEW_PAGES + 1):
        url = (f"https://itunes.apple.com/{APP_COUNTRY}/rss/customerreviews"
               f"/page={page}/id={app_id}/sortby=mosthelpful/json")
        try:
            data    = json.loads(fetch_url(url))
            entries = data.get("feed", {}).get("entry", [])
            if page == 1 and entries:
                entries = entries[1:]
            if not entries:
                break
            for e in entries:
                reviews.append({
                    "review_id":  e.get("id",{}).get("label",""),
                    "stars":      e.get("im:rating",{}).get("label",""),
                    "date":       e.get("updated",{}).get("label","")[:10],
                    "title":      e.get("title",{}).get("label",""),
                    "text":       e.get("content",{}).get("label","").replace("\n"," ").strip(),
                    "source":     "app_store",
                    "product":    BRAND_NAME,
                    "version":    e.get("im:version",{}).get("label",""),
                    "vote_count": e.get("im:voteCount",{}).get("label","0"),
                    "place_name": "", "address": "", "city": "", "state": "",
                    "latitude": "", "longitude": "",
                    "google_rating": "", "total_reviews_at_location": "",
                })
            print(f"   Page {page}: {len(entries)} reviews (total: {len(reviews)})")
            time.sleep(0.5)
        except Exception as ex:
            print(f"   Page {page}: {ex} - stopping.")
            break

    print(f"   ✅ App Store: {len(reviews)} reviews")
    return reviews


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 - Google Maps via SerpAPI
# ══════════════════════════════════════════════════════════════════════════════

def serpapi_get(params):
    params["api_key"] = SERPAPI_KEY
    url = f"https://serpapi.com/search?{urllib.parse.urlencode(params)}"
    return json.loads(fetch_url(url))


def scrape_location_reviews(keyword, max_locations=1, max_pages=1):
    """
    Search Google Maps for keyword, scrape reviews with full location metadata.
    Returns (reviews_list, api_calls_made).
    """
    reviews = []
    calls = 0
    try:
        data    = serpapi_get({"engine": "google_maps", "q": keyword, "type": "search"})
        calls += 1
        results = data.get("local_results", [])

        if not results:
            return [], calls

        for place in results[:max_locations]:
            data_id       = place.get("data_id", "")
            place_name    = place.get("title", keyword)
            raw_address   = place.get("address", "")

            gps  = place.get("gps_coordinates") or {}
            lat  = gps.get("latitude")  or place.get("latitude")  or place.get("lat") or ""
            lon  = gps.get("longitude") or place.get("longitude") or place.get("lng") or place.get("lon") or ""

            google_rating = place.get("rating", "")
            total_reviews = place.get("reviews", "")

            city, state = parse_address(raw_address, search_term=keyword)

            if not data_id:
                continue

            next_token = None
            location_reviews = []

            for page in range(max_pages):
                params = {"engine":"google_maps_reviews","data_id":data_id,"sort_by":"newestFirst","hl":"en"}
                if next_token:
                    params["next_page_token"] = next_token

                try:
                    rdata = serpapi_get(params)
                    calls += 1
                    raw   = rdata.get("reviews", [])
                    if not raw:
                        break

                    for r in raw:
                        text = r.get("snippet", "").replace("\n", " ").strip()
                        if not text:
                            continue
                        location_reviews.append({
                            "review_id":  r.get("review_id", f"{data_id}_{len(location_reviews)}"),
                            "stars":      str(r.get("rating", "")),
                            "date":       parse_relative_date(r.get("date", "")),
                            "title":      "",
                            "text":       text,
                            "source":     "google_maps",
                            "product":    place_name,
                            "version":    "",
                            "vote_count": str(r.get("likes", 0)),
                            "place_name": place_name,
                            "address":    raw_address,
                            "city":       city,
                            "state":      state,
                            "latitude":   str(lat),
                            "longitude":  str(lon),
                            "google_rating":              str(google_rating),
                            "total_reviews_at_location":  str(total_reviews),
                        })

                    next_token = rdata.get("serpapi_pagination", {}).get("next_page_token", "")
                    if not next_token:
                        break
                    time.sleep(0.3)

                except Exception:
                    break

            if location_reviews:
                print(f"      {place_name} ({city}, {state}): {len(location_reviews)} reviews")
            reviews.extend(location_reviews)
            time.sleep(0.3)

    except Exception as ex:
        print(f"   Error for '{keyword}': {ex}")

    return reviews, calls


def scrape_serpapi():
    if not SERPAPI_KEY:
        print("\n🌍 SERPAPI_KEY not set - skipping Google Maps.")
        return []

    print(f"\n🌍 Scraping Google Maps reviews for: {BRAND_NAME} (nationwide)...")
    all_reviews = []
    seen_ids    = set()

    search_terms = []
    for kw in KEYWORDS:
        for state in US_STATES:
            search_terms.append(f"{kw} {state}")

    print(f"   Searching state-targeted queries (capped for API budget)...")

    queries_run    = 0
    max_queries    = 12
    api_calls_used = 0
    max_api_calls  = 90

    for term in search_terms:
        if queries_run >= max_queries:
            break
        if api_calls_used >= max_api_calls:
            print(f"   ⚠️  Approaching API budget limit ({api_calls_used} calls) - stopping early.")
            break
        if len(all_reviews) >= 300:
            break

        reviews, calls_made = scrape_location_reviews(term, max_locations=1, max_pages=1)
        queries_run    += 1
        api_calls_used += calls_made

        for r in reviews:
            if r["review_id"] not in seen_ids:
                seen_ids.add(r["review_id"])
                all_reviews.append(r)

        time.sleep(0.3)

    print(f"\n   ✅ Google Maps: {len(all_reviews)} unique reviews from {queries_run} queries ({api_calls_used} API calls used)")

    if all_reviews:
        states_found = set(r["state"] for r in all_reviews if r["state"])
        print(f"   📍 States covered: {sorted(states_found)}")

    return all_reviews


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def save_reviews(reviews):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REVIEWS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(reviews)
    print(f"\n   💾 Saved {len(reviews)} reviews → {REVIEWS_CSV}")


def main():
    print("=" * 55)
    print(f"  Smart Data Extractor - {BRAND_NAME}")
    print("  Combines App Store + Google Maps automatically")
    print("=" * 55)

    all_reviews = []

    app_reviews = scrape_app_store()
    all_reviews.extend(app_reviews)

    maps_reviews = scrape_serpapi()
    all_reviews.extend(maps_reviews)

    if not all_reviews:
        print("\n⚠️  No reviews collected from any source.")
        print("   Check APP_STORE_ID and SERPAPI_KEY in config.py / GitHub Secrets")
        sys.exit(1)

    save_reviews(all_reviews)

    sources = {}
    for r in all_reviews:
        src = r.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\n" + "=" * 55)
    print(f"  ✅ Done - {len(all_reviews)} total reviews")
    for src, count in sources.items():
        print(f"     {src}: {count}")
    print("=" * 55)


if __name__ == "__main__":
    main()
