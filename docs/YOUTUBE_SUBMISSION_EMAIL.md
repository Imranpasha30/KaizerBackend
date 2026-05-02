# YouTube API Compliance Review — Reply Email

> **For the senior submitter.** Paste the body below verbatim into the
> reply to YouTube's compliance email. Then attach the doc described
> at the bottom.

---

## TO: replied to the existing compliance-review thread (use Reply, do NOT compose new)

## SUBJECT: keep the existing subject line YouTube assigned (do not change)

## BODY

```
Hi YouTube API Compliance Review team,

Thank you for the follow-up on our quota increase request for Kaizer
News. Please find below the requested information, screen-cast
recordings, and a link to our full compliance write-up.

— Application identity
   Project / API client name:  Kaizer News
   OAuth client ID:            542271243369-6ikvqv149ht0s569u4prf11kkm8vqalh.apps.googleusercontent.com
   Production frontend:        https://kaizerx.com
   Production backend:         https://api.kaizerx.com
   Privacy policy:             https://kaizerx.com/privacy
   Terms of service:           https://kaizerx.com/terms

— Organization-owned YouTube channels (videos are uploaded to these)
   1. Auto Wala       — <FILL: https://www.youtube.com/@…>  — UC<FILL>
   2. Cyber Sphere    — <FILL: https://www.youtube.com/@…>  — UC<FILL>
   3. Kaizer Upload   — <FILL: https://www.youtube.com/@…>  — UC<FILL>

— Screen-cast recordings (Google Drive, anyone-with-the-link viewing)

   Video 1 — Organization-channel video management:
     https://drive.google.com/file/d/1_FFV2nOme9wFtjojYuRpm4Z9cSRN7Voq/view?usp=sharing

   This recording shows the org admin selecting a generated clip,
   opening the Publish modal, picking the destination YouTube channel
   from the connected list, and clicking Publish. Behind the scenes
   our backend creates an UploadJob row, mints a fresh access token
   from the encrypted refresh token, applies the channel's logo
   overlay via FFmpeg, and calls videos.insert using the resumable
   upload protocol.

   The actual upload to YouTube does NOT complete in the recording
   because our YouTube Data API v3 quota is currently exhausted —
   each videos.insert call costs 1,600 units against the default
   10,000/day quota, which caps us at approximately 6 full uploads
   per day across all our channels. This is exactly the constraint
   the requested quota increase will resolve. The same code path has
   been verified working end-to-end in our staging environment at a
   higher quota allocation.

   Video 2 — Public-creator OAuth flow with consent screen:
     https://drive.google.com/file/d/1z6MXUYErunlOO7XbSoaZkmkQeFk6Jq46/view?usp=sharing

   This recording shows the OAuth 2.0 consent flow that any public
   creator (the SaaS user base) goes through to connect their own
   YouTube channel. It uses a Google account that has never
   authorised Kaizer News before, so the full consent screen with
   the scope list is visible.

   The "Google hasn't verified this app" warning shown at the start
   of the recording is the standard advisory Google displays during
   the verification window — we are currently awaiting completion of
   the branding verification associated with this exact compliance
   review. The advisory will clear automatically once verification
   completes. We have not bypassed it in the recording; the user
   clicks Advanced → Go to Kaizer News (unsafe), exactly as Google's
   policy mandates during the verification window.

   The actual scope-listing consent screen is then visible. The user
   clicks Allow; our backend exchanges the code for tokens, encrypts
   the refresh token, captures the channel ID + title, and redirects
   the user back to kaizerx.com where the connected channel name
   appears in the Style Profiles UI.

— OAuth scopes requested (exactly these three, no more)

   • https://www.googleapis.com/auth/youtube.upload     — to call
     videos.insert and upload the user's processed clips.
   • https://www.googleapis.com/auth/youtube.readonly   — to call
     channels.list?mine=true once after the OAuth callback so we can
     show the user "Connected as <channel name>" in the UI.
   • https://www.googleapis.com/auth/youtube            — to call
     thumbnails.set with the auto-generated 9:16 thumbnail and (for
     a planned feature) videos.update if a creator fixes a typo from
     our editor.

   We do NOT request youtube.force-ssl, analytics, partner, members,
   livestream, captions, comments, subscriptions, or any other
   scope. Identity scopes (openid, userinfo.email, userinfo.profile)
   are auto-granted by Google during sign-in but are never used or
   stored by us.

— What we DO NOT do
   - We do not sell, share, rent, or trade YouTube data with anyone.
   - We do not train AI models on user-uploaded videos or channel
     content.
   - We do not access subscribers, comments, view data, analytics,
     or any data beyond the user's own uploaded videos.
   - We do not upload to channels the user has not explicitly
     authorised. Each oauth_tokens row in our database corresponds
     1:1 to a user's individual OAuth grant for one specific channel.

— Revocation
   Users can revoke our access at any time:
   • In-app: the Disconnect button on every connected account card on
     the Style Profiles page calls DELETE /api/youtube/oauth/{id},
     which clears refresh_token_enc.
   • Externally: https://myaccount.google.com/permissions.

— Data security
   - Refresh tokens encrypted at rest with AES-256 (cryptography.fernet)
     using a key stored only in our production environment.
   - Access tokens minted on demand, used for one API call, never
     persisted.
   - All transit over HTTPS / TLS 1.2+.
   - User passwords hashed with bcrypt (cost 12).

A complete technical write-up — every YouTube API method we call, the
full OAuth flow, data security details, multi-tenant isolation, and
pre-empted answers to common reviewer questions — is attached as a
PDF (YOUTUBE_COMPLIANCE_REVIEW.pdf).

Please let us know if you need anything further.

Thanks,
<Senior submitter — fill in name and role here>
Kaizer News
contact: devsharkify@gmail.com
```

---

## Before sending — fill these 3 placeholders

The body above contains exactly three things the senior must fill in
before sending:

1. **Channel URLs + UC IDs** in the channel-list section (3 rows). Get
   the UC IDs by opening each channel on YouTube → About → Share
   channel → Copy channel ID. Format: `UCxxxxxxxx...`.
2. **The senior's name + role** at the bottom (`<Senior submitter — fill in name and role here>`).
3. Confirm the contact email (`devsharkify@gmail.com`) is the one the
   reviewer should use, or replace with a different one.

---

## What to attach with the email

1. **`YOUTUBE_COMPLIANCE_REVIEW.md` exported as PDF** — the full
   technical write-up at
   `kaizer/KaizerBackend/docs/YOUTUBE_COMPLIANCE_REVIEW.md`. Export
   to PDF from VS Code's Markdown preview (right-click → "Markdown
   PDF: Export (pdf)") OR via Pandoc:
   ```
   pandoc YOUTUBE_COMPLIANCE_REVIEW.md -o YOUTUBE_COMPLIANCE_REVIEW.pdf
   ```
   Reviewers prefer PDF over .md attachments.

That's the only attachment needed — everything else (videos, URLs,
client ID) is inline in the email body.

---

## Send checklist for senior

- [ ] All 3 channel URL + UC ID placeholders replaced with real values.
- [ ] Senior's name + role filled in at the bottom.
- [ ] PDF of `YOUTUBE_COMPLIANCE_REVIEW.md` attached to the email.
- [ ] Both Google Drive video links **tested in incognito** before
      sending — confirm "Anyone with the link can view" is set on
      each video, not "Restricted".
- [ ] Email sent as a **Reply** to YouTube's existing compliance
      thread (not a new compose), so the original ticket reference
      is preserved.

---

## What to expect after sending

YouTube's policy is to respond within **7 business days**. Outcomes:

1. **Approved** — quota increase is granted; the unverified-app
   warning on the OAuth consent screen disappears at the same time.
2. **Follow-up questions** — see Section 9 of the compliance doc;
   most likely topics are pre-empted there. Quote the relevant
   answer back to them in the reply.
3. **Rejected with a reason** — the rejection email cites the
   specific gap. Fix that one item, do not re-record everything,
   and resubmit.
