# Homepage Compliance Audit — Kaizer News

> Audit of `https://kaizerx.com/` against Google's six homepage
> requirements for OAuth verification. Each row says **what Google
> wants**, **what kaizerx.com currently shows**, and **what to do**
> if there's still a gap.

Last updated: 2026-04-29.

---

## The 6 requirements + our status

### 1. Accurately represent and identify your app or brand

| Google's requirement | Our status |
|---|---|
| Homepage clearly identifies the app | ✅ Hero shows "KAIZER NEWS" logo + brand |
| App name matches what's on the OAuth consent screen | ✅ "Kaizer News" matches Cloud Console branding |
| Logo on homepage matches the consent-screen logo | ✅ Same K logo on both |

**Action:** none.

### 2. Fully describe your app's functionality to users

| Google's requirement | Our status |
|---|---|
| Plain-language description of what the product does | ✅ Hero copy: "Go live with zero operators. Clip your archive while you sleep." |
| Detailed feature explanation visible without login | ✅ Product section, features grid, demo video, Live Director deep-dive — all on the public landing page |

**Action:** none. The landing page already does this.

### 3. Explain with transparency the purpose for which your app requests user data ⚠️ **GAP — fixed by this audit**

| Google's requirement | Our status |
|---|---|
| Public-facing section explaining what YouTube data is requested | ❌ Missing — was the most likely next rejection reason |
| Map each requested scope to a specific feature | ❌ Missing |
| Statement that user data isn't sold / shared / used to train AI | ❌ Was only in the Privacy Policy, not on the home page |

**Action:** added a **"How we use your YouTube data"** section directly on the landing page (visible without login) — see commit `<this commit>` to `pages/Landing.jsx`. It explicitly:

  - Names the three scopes we request
  - Describes what each scope unlocks (uploading clips, reading channel name, setting custom thumbnails)
  - States we don't use customer videos to train AI
  - Links to the privacy policy + terms
  - Mentions the user can revoke access at any time via Google account permissions

This addresses the gap **before** Google flags it as a follow-up rejection after the privacy-link issue clears.

### 4. Hosted on a verified domain you own

| Google's requirement | Our status |
|---|---|
| Domain ownership verified | ✅ `kaizerx.com` verified in Google Search Console |
| Domain matches what's on the OAuth consent screen | ✅ Cloud Console → Branding lists `kaizerx.com` as the home page URL |
| Not on a third-party platform that doesn't allow subdomain ownership | ✅ Custom domain on Railway, not Google Sites / Facebook / Twitter |

**Action:** none.

### 5. Include a link to your privacy policy

| Google's requirement | Our status |
|---|---|
| Privacy policy link visible on the home page | ✅ Footer "Privacy" link points to `/privacy` (commits `a160dfe`, `1a12b12`) |
| Static fallback for crawlers that don't run JS | ✅ Static `<a href="/privacy">` baked into `index.html` outside the React mount — verified with `view-source:` |
| Link target matches the URL in Cloud Console branding | ✅ Both point at `https://kaizerx.com/privacy` |

**Action:** none. Verified live: `https://kaizerx.com/` raw HTML contains `href="/privacy"` twice.

### 6. Visible to users without requiring them to log-in

| Google's requirement | Our status |
|---|---|
| Home page itself is public | ✅ `Route path="/"` in `App.jsx` is NOT wrapped in `ProtectedRoute` |
| Privacy + terms pages are public | ✅ `/privacy` and `/terms` are explicitly listed in `Shell.hideChrome` and unwrapped in `App.jsx` |
| Description of the app is readable without login | ✅ Hero, features, product, pricing — all rendered at `/` without auth |

**Action:** none.

---

## Common rejection reasons — our defence against each

| Rejection reason | Why it could hit us | Mitigation |
|---|---|---|
| **Unresponsive homepage URL** | Railway custom domain misconfigured | Verified live: `https://kaizerx.com/` returns 200 with `<title>KAIZER NEWS — Autonomous Live Director & AI Clips</title>`. DNS via Cloudflare CNAME (proxied), Railway holds the cert. |
| **Homepage not registered to you** | Domain ownership not verified in Search Console | ✅ User completed Search Console domain verification (DNS TXT). Same Google account that owns the OAuth project. |
| **Homepage redirects to a different domain** | A registrar-level forwarder pointing kaizerx.com → some other URL | None — kaizerx.com is served directly by Railway. No redirects in the chain. |
| **Shortened / condensed link** | Using bit.ly / g.co style URL | None — `https://kaizerx.com` is a clean root domain. |
| **Homepage behind login** | Landing route protected | None — `Route path="/"` is public. Tested in incognito. |
| **Homepage doesn't display app information** | Empty / placeholder hero | None — the landing page has a hero, product description, features grid, demo video, pricing, testimonials. |
| **Homepage doesn't link to privacy policy** | Footer link missing or broken | ✅ Three places: (1) static `<footer>` in `index.html`, (2) `<noscript>` block, (3) React-rendered Landing footer. Crawlable in raw HTML and visible to users. |

---

## How the verification flow actually works

1. You add app info + URLs in **Branding** tab.
2. You click **Save** — Google's verifier runs once, finds issues if any.
3. **You fix the issues** (deploy code changes, add DNS records, etc.).
4. **You click "I have fixed the issues" → Proceed** ← *this triggers a re-crawl*.
5. Google re-runs the verifier on the same URLs.
6. Either the branding badge goes ✓ or you get a NEW (more specific) issue.

**Critical point:** step 4 must happen **after** every fix. Google does NOT re-verify automatically; it shows you the cached result of your last "Proceed" click. If you fix something but don't click Proceed, you'll keep seeing the same stale rejection forever.

---

## Order of operations from here

1. **Confirm `https://kaizerx.com/` shows the privacy footer link** in incognito. *(Already verified — done.)*
2. **Confirm `https://kaizerx.com/privacy` loads in incognito.** *(Already verified — done.)*
3. **Wait for Railway to redeploy the new "How we use your YouTube data" section** (this commit). *(2-3 min after push.)*
4. **Hard-refresh `https://kaizerx.com/`** in incognito. Scroll past the hero — you should see a section titled "How we use your YouTube data" with three scope rows.
5. **Click "I have fixed the issues" → Proceed** in Cloud Console → Branding → View issues.
6. **Wait** for re-verification (5-30 min).
7. **If rejected with a NEW reason**, that means the privacy-link issue cleared and we're onto the next one — send the screenshot, I'll fix it.
8. **If accepted**, branding badge goes ✓. Move on to **Audience** tab → confirm "In production". Then **Verification Center** for the final OAuth scope review (which is what unlocks the YouTube quota increase).

---

## What the YouTube quota review is actually checking

Branding verification ≠ YouTube quota review. They're separate processes.

| Process | What it checks | When it runs |
|---|---|---|
| **Branding verification** (where we are now) | Home page, logo, privacy/terms URLs, domain ownership | Triggered by you clicking "I have fixed the issues" |
| **OAuth scope verification** (next) | Each requested scope is justified, your app's data handling is compliant | Triggered when you submit for verification with sensitive/restricted scopes |
| **YouTube API Services quota review** (the original ticket) | YouTube-specific compliance — videos uploaded to org-owned channels, OAuth flow for public users, API method usage | The thing you originally got the email about; uses the screen-cast videos |

Branding is a prerequisite for the others. Once it clears, the others unblock.
