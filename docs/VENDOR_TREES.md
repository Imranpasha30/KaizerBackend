# Vendor trees (read-only port sources) — RULES

Sibling repos cloned into `e:\kaizer-dev\` as **read-only reference/vendor trees**.
We PORT code from them; we NEVER import from them, never add them to sys.path,
never wire them into builds. Every ported file carries a header comment:
`# Ported from kaizer-platform@<sha> <original path>`.

| Tree | Origin | State |
|---|---|---|
| `e:\kaizer-dev\KaizerBackendendor\kaizer-platform` | github.com/devsharkify/kaizer-platform | snapshot @ `d5fd482` (2026-07-18, shallow depth-1 — repo went private before a full clone; re-clone with history once the operator grants access to Imranpasha30 or provides a devsharkify token) |
| `e:\kaizer-dev\avatar-studio` | github.com/devsharkify/avatar-studio (PRIVATE) | NOT YET CLONED — operator must grant access. Required for the MuseTalk/IndicF5 news-anchor path (`AVATAR_STUDIO_DIR`); until then the anchor feature runs on the HeyGen provider. |
| `e:\kaizer-dev\bulletin-gen` | github.com/devsharkify/bulletin-gen (private/unknown) | NOT YET CLONED — needed only for the newsroom-autopilot job types; deferred. |

Port targets already planned: `pipeline_v4/director_platform/` (their ai_director),
`pipeline_v4/svg_template.py` (their SVG slot templates), `avatar/` + `pipeline_core/podcast/`
+ `e:\kaizer-dev\remotion\` (news-anchor stack), `routers/desktop.py` + `DesktopLicense`
(desktop auth/licensing), desktop Electron scaffold (`e:\kaizer-dev\kaizer-desktop`).
