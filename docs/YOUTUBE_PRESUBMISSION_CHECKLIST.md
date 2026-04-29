# YouTube API Quota Review — Pre-Submission Checklist

> **Do every step below in order before you click submit.** Each row is a
> common rejection reason. The detailed write-up
> (`YOUTUBE_COMPLIANCE_REVIEW.md`) and the two screen-cast videos
> won't matter if any of these items is wrong, because the reviewer
> stops at the first failure.

Estimated time: **2-3 hours of careful setup**, then ~30-45 minutes
of recording, then submit.

---

## Step 1 — Verify Cloud Console hygiene  *(20 min)*

Go to https://console.cloud.google.com → select your project →
**APIs & Services → OAuth consent screen**.

- [ ] **App name** is exactly `Kaizer News`. (No version numbers, no
      "test", no emoji.)
- [ ] **User support email** is a real address you check.
- [ ] **Application logo** is uploaded — at least 120×120 PNG, square.
      Use your brand mark; this is the logo every creator sees on
      Google's consent screen.
- [ ] **Application home page** = `https://kaizerx.com` (or whatever
      the current production frontend URL is).
- [ ] **Application privacy policy link** = a URL that **actually
      loads** when you paste it in incognito. *(See Step 2.)*
- [ ] **Application terms of service link** = a URL that **actually
      loads** in incognito. *(See Step 2.)*
- [ ] **Authorised domains** lists every domain that appears in any
      redirect URI:
      - `kaizerx.com`
      - `up.railway.app`  *(needed because your backend redirect URI
        is on the railway subdomain)*
- [ ] **Publishing status = "In production"**, NOT "Testing".
      *(Click "Publish App" if it still says Testing.)*

Then **APIs & Services → Credentials → click your OAuth Web Client**:

- [ ] **Authorised redirect URIs** includes exactly the URI your
      backend uses:
      `https://kaizerbackend-production.up.railway.app/api/youtube/auth/callback`
- [ ] **Authorised JavaScript origins** includes the frontend URL(s).

---

## Step 2 — Privacy policy + Terms of Service  *(45 min if you have none)*

This is the **#1 silent killer**. Reviewers paste the URLs from your
Cloud Console listing and walk away the moment one returns 404.

- [ ] Privacy policy is published at a stable URL (e.g.
      `https://kaizerx.com/privacy`).
- [ ] Terms of service is published at a stable URL (e.g.
      `https://kaizerx.com/terms`).
- [ ] Both URLs return HTTP 200 in **incognito** (not just to logged-in
      users).
- [ ] Privacy policy explicitly mentions:
      - You request the user's permission to upload videos to YouTube.
      - You store an encrypted refresh token.
      - The user can revoke access at any time.
      - You comply with the YouTube API Services Terms of Service and
        the Google Privacy Policy.
- [ ] The privacy policy contains a link to:
      - **YouTube Terms of Service** —
        `https://www.youtube.com/t/terms`
      - **Google Privacy Policy** —
        `https://policies.google.com/privacy`

If you don't have a written one yet: generate a draft on
**privacypolicies.com** or **termly.io** (both free tiers exist),
review it, edit in the YouTube-specific clauses above, and publish.

---

## Step 3 — Domain verification  *(15 min)*

- [ ] In **Google Search Console** (search.google.com/search-console)
      verify ownership of `kaizerx.com` (DNS TXT record method is
      the most reliable).
- [ ] Verify ownership of any other domain that appears in your
      OAuth consent screen's "Authorised domains".
- [ ] Confirm in Cloud Console → OAuth consent screen → "Authorised
      domains" the verified domains show without warning.

---

## Step 4 — Brand sanity check  *(10 min)*

The reviewer cross-references your consent screen against the docs +
videos. Inconsistencies trigger flags.

- [ ] App name on consent screen = "Kaizer News".
- [ ] App name on your website's `<title>` and homepage hero =
      "Kaizer News" (not a placeholder).
- [ ] Privacy policy header / page title says "Kaizer News".
- [ ] Logo on the consent screen matches the logo on your site nav.

If you have a different working name (e.g. "Ozone Wash") plan to
serve a different audience, **don't ship that branding to the
YouTube reviewer**. The OAuth client must look unambiguous. Either
rename the OAuth client or rename the public site to match.

---

## Step 5 — Test the OAuth flow yourself end-to-end  *(20 min)*

- [ ] Open an incognito window.
- [ ] Use a Google account that has **never** authorised Kaizer News.
      (If unsure, revoke first at
      https://myaccount.google.com/permissions.)
- [ ] Go through the entire flow: register → create profile →
      Link my YouTube → grant consent → see the connected channel
      title in our UI.
- [ ] Upload a sample video, generate clips, publish one clip to the
      newly-connected channel, confirm it appears in the YouTube
      Studio for that channel.
- [ ] Click Disconnect in our UI, confirm we no longer mint access
      tokens for that channel.

If any step here fails, the reviewer's identical attempt will fail
too. Fix before recording.

---

## Step 6 — Record the two screen-cast videos  *(45 min)*

Follow the shot-by-shot script in
**`YOUTUBE_COMPLIANCE_REVIEW.md` → Section 8**.

Setup:

- [ ] Recording resolution **1280×720 minimum** (1920×1080 is better).
- [ ] Audio: clear English voice-over, no background music.
- [ ] No notifications popping up — Do Not Disturb mode on.
- [ ] DevTools closed unless explicitly demonstrated. If shown, do
      not display secrets, JWTs, or environment variables.
- [ ] Use a **brand-new test Google account** for Video 2 so the
      consent screen with the scope list is visible.

Video 1 — organisation channels  *(~3 min)*:

- [ ] Sign in as the org admin user.
- [ ] Walk through Style Profiles showing the connected channels.
- [ ] Upload a real long-form video, let the pipeline run.
- [ ] Open the Publish modal, pick all org channels, publish.
- [ ] Show `/uploads` page progressing through queued → uploading →
      done.
- [ ] Open one of the uploaded videos in YouTube Studio and confirm
      title/description/tags are correct.

Video 2 — public-creator OAuth flow  *(~4 min)*:

- [ ] Register a new account in incognito.
- [ ] Click **Link my YouTube** on the new profile.
- [ ] **Pause for 3-5 seconds on Google's consent screen** so the
      reviewer can read the three scope lines.
- [ ] Grant consent, show the connected channel name in our UI.
- [ ] Upload a video, generate clips, publish to the channel.
- [ ] Show the live result on YouTube.
- [ ] Demonstrate revocation via myaccount.google.com/permissions.

After recording:

- [ ] Upload both videos to YouTube as **Unlisted** under the
      organisation account (not the test account).
- [ ] Watch each video back end-to-end one time. If any moment is
      blurry, missing audio, or unclear → re-record that section.
- [ ] Copy the two YouTube watch URLs.

---

## Step 7 — Fill in the doc placeholders  *(10 min)*

Open `YOUTUBE_COMPLIANCE_REVIEW.md` and replace the placeholders:

- [ ] **Section 1** — Privacy policy URL, ToS URL.
- [ ] **Section 1** — Channel list table: replace the three example
      rows with your real `https://www.youtube.com/@…` URLs and
      the corresponding `UC...` channel IDs.
- [ ] **Section 9** — OAuth client ID (from Cloud Console →
      Credentials → your OAuth Web Client).
- [ ] **Section 9** — Both video URLs.

Save the file. This is what you'll attach (or paste from) in your
reply to YouTube.

---

## Step 8 — Reply to YouTube  *(10 min)*

In the reply to the email you received:

- [ ] Use the exact email-template wording from
      `YOUTUBE_COMPLIANCE_REVIEW.md` Section 9.
- [ ] Attach `YOUTUBE_COMPLIANCE_REVIEW.md` as a PDF (export from
      VS Code's Markdown preview or via Pandoc) — many reviewers
      prefer PDF over .md.
- [ ] Paste the two YouTube video URLs in-line (not just as
      attachments).
- [ ] Send.

---

## After you submit

- They are required to respond within seven business days.
- If they ask follow-up questions, the most likely topics are
  pre-empted in `YOUTUBE_COMPLIANCE_REVIEW.md` Section 10. Quote
  the relevant answer back to them.
- If they reject, the rejection email tells you exactly what was
  missing — fix that, re-record only that segment of the video,
  resubmit. Don't re-record the whole thing.

---

## Things that will get you rejected even if everything above is right

- App is in **Testing** mode, not Production. *(Step 1.)*
- Privacy policy or ToS URL returns 404. *(Step 2.)*
- Consent-screen logo missing. *(Step 1.)*
- Video skips or blurs the consent screen. *(Step 6.)*
- Video uses a Google account that previously authorised the app —
  consent screen short-circuits past the scope list. *(Step 6.)*
- Brand inconsistency between consent screen, website, privacy
  policy. *(Step 4.)*

If you triple-check those six and the rest of the checklist, this
will pass.
