"""Self-contained, printable weekly model packets using existing team assets."""
import base64
import io
import json
import re
from functools import lru_cache
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

import model_spec


STYLE = '''
body{font:15px/1.42 system-ui,sans-serif;color:#203039;background:#f1f3f1;margin:0}
main{max-width:1050px;margin:auto;padding:22px 20px}h1,h2,h3{font-weight:600;line-height:1.2}
h1{font-size:34px;margin:4px 0 2px}h2{font-size:24px}h3{font-size:17px}.eyebrow{letter-spacing:2px;text-transform:uppercase;font-size:12px;color:#64796f}
.muted,small{color:#63716d}.card{background:white;border:1px solid #dce3dd;border-radius:10px;padding:16px 18px;margin:12px 0;break-inside:avoid}
.teams{display:flex;align-items:center;gap:15px}.logo{width:52px;height:52px;object-fit:contain}.metrics{display:flex;flex-wrap:wrap;gap:20px;margin:10px 0}.metric strong{display:block;font-size:25px;font-weight:550}
.pill{display:inline-block;background:#edf1ec;border-radius:4px;padding:4px 9px;font-size:12px;font-weight:600}.warn{border-left:3px solid #b18b4f;padding:10px 16px;background:#faf7f0}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px 8px;text-align:right;border-bottom:1px solid #edf0ed}td:first-child,th:first-child{text-align:left}a{color:#236950}
.columns{display:grid;grid-template-columns:1.2fr 1fr;gap:20px}.barrow{display:grid;grid-template-columns:180px 1fr 55px;align-items:center;gap:8px;font-size:12px;margin:4px 0}.track{position:relative;height:12px;background:linear-gradient(90deg,#f7f4ef 50%,#eef4f0 50%)}.bar{position:absolute;height:12px;background:#30765d}.negative{background:#b17f51}.value{text-align:right;font-variant-numeric:tabular-nums}
/* Full feature list, not just the top 12 -- scrolls in place instead of
   turning the whole page into one long bar chart. */
.importance-scroll{max-height:480px;overflow-y:auto;padding-right:6px;border:1px solid #ffffff12;border-radius:6px}
.importance-scroll .barrow{margin:7px 10px}
@media(max-width:760px){.columns{grid-template-columns:1fr}.barrow{grid-template-columns:145px 1fr 45px}main{padding:12px 10px}}
@media print{body{background:white}main{padding:0}.card{border-radius:0;page-break-inside:avoid}a{color:inherit}.no-print{display:none}}
'''

STYLE += '''
body{font-family:Arial,sans-serif;background:white;color:#222}
main{max-width:1150px;padding:16px}h1{font-size:26px}h2{font-size:21px}
.card{border:2px solid #aaa;border-radius:0;padding:14px;margin:12px 0}
td,th{border:1px solid #bbb;padding:5px 9px}th{background:#eee}
.pill{border-radius:0}.eyebrow{letter-spacing:0}.columns{display:block}
.rank{color:#666;font-size:12px;margin-left:8px}.contribution{position:relative;height:26px;background:linear-gradient(90deg,#f5eee6 50%,#edf4ef 50%)}
.contribution:after{content:"";position:absolute;left:50%;height:100%;border-left:1px solid #aaa}
.contribution .bar{height:26px;opacity:.45}.contribution b{position:relative;z-index:1;display:block;text-align:center;line-height:26px;font-size:12px}
.reconcile{border-top:2px solid #999;padding-top:10px;text-align:right}
details{margin:8px 0;font-size:13px;color:#666}summary{cursor:pointer}
.packet{max-width:800px;background:#111;color:#eee}
body:has(.packet){background:#111}
.packet .card{border:0;border-top:1px solid #353535;background:transparent;padding:14px 0}
.packet .muted,.packet small,.packet details,.packet .rank{color:#aaa}
.packet a{color:#8bb9ff}.packet .pill{background:#292929;color:#ccc}
.packet .warn{background:#28231b;color:#eedbb9}
.packet-tabs{display:flex;gap:6px;border-bottom:1px solid #444;margin:0 0 12px;position:sticky;top:0;background:#111;z-index:5;padding:6px 0}
.packet-tabs a{padding:8px 12px;text-decoration:none;color:#aaa;font-size:14px}
.packet-tabs a[aria-current="page"]{color:white;border-bottom:3px solid #8bb9ff}
/* bundle_single_file()'s CSS-only tabs -- no JS, so it still works once
   detached from the folder and opened as one standalone file. Radios and
   panels are flat siblings of .packet-bundle (flex + order stacks the
   tab row on top of whichever panel is checked); the label being the
   radio's literal next sibling is what lets :checked style it directly. */
.packet-bundle{display:flex;flex-wrap:wrap;background:#111;min-height:100vh}
.packet-bundle .pkgtab-radio{display:none}
.packet-bundle .pkgtab-label{order:1;flex:0 0 auto;padding:14px 16px;cursor:pointer;color:#aaa;font-size:14px;border-bottom:3px solid transparent}
.packet-bundle .pkgtab-radio:checked+.pkgtab-label{color:white;border-color:#8bb9ff}
main.pkgpanel{order:2;flex:1 0 100%;display:none}
#pt-headline:checked~.pkgpanel[data-tab=headline]{display:block}
#pt-spread:checked~.pkgpanel[data-tab=spread]{display:block}
#pt-total:checked~.pkgpanel[data-tab=total]{display:block}
#pt-importance:checked~.pkgpanel[data-tab=importance]{display:block}
.headline-frame{width:100%;height:80vh;border:0;background:white}
main.headline-shell{max-width:1500px}
/* The sheet sizes itself to its columns (.headline-table{width:auto}), so
   a 1500px shell left several hundred pixels of empty page beside it.
   Only the headline tab gets pulled in -- the stats tab shares
   .headline-shell and genuinely needs the room. 1000px is the sheet's
   width (~846px measured at week 3 with 1px 4px cells; 2px 6px adds
   ~2px a side to most of its columns) plus headroom for longer
   quarterback names; anything wider still overflows to a scrollbar
   rather than being clipped, via .headline-shell's overflow-x. */
main.sheet-page{max-width:1000px}
.packet th{background:#222}.packet td,.packet th{border-color:#333}
.matchup{margin:4px 0 8px}
/* .matchup-head (the header/summary of each collapsible section) and
   .stat-line (each metric row inside it) MUST land their bar tracks at the
   identical x-position/width so the whole card's bars stack in one visual
   column -- --gutter/--val-w/--gap are the single source of truth for
   that; if you touch one you touch all three grid-template-columns below.
   matchup-head: [gutter][bar][gutter]. stat-line: [gutter-val-gap][val]
   [bar][gutter] -- the label+value on the left sum to the same width as
   one gutter, so the bar's left edge lines up with matchup-head's; the
   right column is a bare gutter so the right edge lines up too. */
.packet{--gutter:200px;--val-w:64px;--row-gap:8px}
.stat-line{display:grid;grid-template-columns:calc(var(--gutter) - var(--val-w) - var(--row-gap)) var(--val-w) minmax(0,1fr) var(--gutter);align-items:center;gap:var(--row-gap);font-size:13px}
.matchup-head{display:grid;grid-template-columns:var(--gutter) minmax(0,1fr) var(--gutter);align-items:center;gap:var(--row-gap);margin-bottom:5px;font-size:15px;font-weight:600}
.matchup-head .side{font-size:13px;display:flex;align-items:center;gap:6px;min-width:0}
.matchup-head .side:last-child{justify-content:flex-end}
.matchup-head .side img.logo{flex:none;display:inline-block;width:22px;height:22px;vertical-align:middle}
/* Ellipsis lives on this inner span, never on .side itself -- .side mixes
   an <img> in with the text, and text-overflow on a box with a non-text
   sibling doesn't reliably ellipsis at the right edge (it was hard-clipping
   from the wrong side and colliding with the logo). Isolated to just the
   text node, it truncates predictably at the text's own end. */
.matchup-head .side .side-text{flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* Grid/flex lives on a plain div (.matchup-head) nested inside <summary>,
   not on <summary> itself -- summary has inconsistent cross-browser support
   for grid/flex on its own box, safest to keep it a bare disclosure trigger. */
summary.matchup-summary{cursor:pointer;list-style:none;position:relative;padding-left:16px}
summary.matchup-summary::-webkit-details-marker{display:none}
summary.matchup-summary:before{content:"▸";position:absolute;left:0;top:50%;transform:translateY(-50%);color:#888;font-size:12px}
details.matchup[open] summary.matchup-summary:before{content:"▾"}
details.matchup{position:relative}
.stat-row{padding:1px 0;border-bottom:1px solid #ffffff09}
.stat-name{text-align:left;font-size:11px;margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat-number{font-variant-numeric:tabular-nums;white-space:nowrap}
.stat-number:last-child{text-align:right}.stat-number .rank{display:inline;margin-left:6px;font-size:11px}
.packet .contribution{height:15px;background:#ffffff05}
.packet .contribution:after{top:-4px;height:calc(100% + 8px);border-color:#ddd}
.matchup-head .net-bar{margin:0}
.packet .contribution .bar{height:15px;opacity:1;box-shadow:inset 0 0 0 1px #ffffff35}
.packet .contribution b{font-size:10px;line-height:15px;color:white;text-shadow:0 1px 2px black,0 0 3px black}
/* Net bar (the header's own summary bar) reads slightly bigger than the
   per-metric detail bars beneath it -- it's the headline number for the
   section, the details are supporting evidence. */
.matchup-head .net-bar .contribution,.matchup-head .net-bar .contribution .bar{height:20px}
.matchup-head .net-bar .contribution:after{top:-4px;height:calc(100% + 8px)}
.matchup-head .net-bar .contribution b{font-size:11px;line-height:20px}
.packet .card{padding:8px 0}.packet .metrics{margin:8px 0;gap:18px}
.stat-bars{display:flex;gap:8px;height:5px;margin-top:8px}
.stat-bars span{border-radius:4px;min-width:0}
.stat-bars .left{background:var(--left-color,#888)}.stat-bars .right{background:var(--right-color,#888)}
.stat-bars span{box-shadow:0 0 0 1px #ffffff40}
.stat-impact{text-align:center;font-size:12px;color:#aaa;margin-top:4px}
.stat-impact.left,.stat-impact.right{color:#ddd}
.stat-impact:before{content:"";display:inline-block;width:9px;height:9px;margin-right:6px;border-radius:50%;box-shadow:0 0 0 1px #ffffff60}
.stat-impact.left:before{background:var(--left-color,#888)}.stat-impact.right:before{background:var(--right-color,#888)}
.packet .reconcile{font-size:14px;border-color:#444}
.game-heading{display:flex;align-items:center;gap:9px;margin:0 0 10px}
.game-heading .logo{width:34px;height:34px}.game-heading h2{margin:0;font-size:20px}
.game-heading .pill{margin-left:auto;font-size:10px}
.game-line{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:14px;padding-bottom:12px;border-bottom:1px solid #333}
.game-line span{white-space:nowrap}.game-line small{margin-right:5px}
.scoreboard{display:flex;justify-content:center;align-items:center;gap:0;margin:5px 0;font-variant-numeric:tabular-nums}
.scoreboard .score-team{padding:5px 9px;border-radius:2px;font-size:16px;font-weight:700;color:white;text-shadow:0 1px 2px #000}
.scoreboard .score{padding:0 9px;font-size:27px;font-weight:800;line-height:1.15}
.scoreboard .score-dash{color:#aaa;font-size:15px;padding:0 2px}
.qb-line{display:flex;flex-wrap:wrap;gap:8px 24px;color:#bbb;font-size:12px;margin:-2px 0 9px}
.match-banner{display:grid;grid-template-columns:150px minmax(0,1fr) 150px;gap:14px;align-items:center;padding:8px 0 16px;border-bottom:1px solid #333}
.banner-team .identity{display:flex;align-items:center;gap:8px;font-size:25px;font-weight:700}
.banner-team .logo{width:62px;height:62px}.banner-team:last-child{text-align:right}
.banner-team:last-child .identity{justify-content:flex-end}
.banner-qb{font-size:12px;color:#bbb;line-height:1.5}
.banner-center{text-align:center}.banner-center .game-line{justify-content:center;border:0;padding:7px 0;gap:8px 14px;font-size:13px}
.banner-total{font-size:12px;color:#ccc;margin-top:6px}
@media(max-width:650px){.match-banner{grid-template-columns:1fr 1fr}.banner-center{grid-column:1/-1;grid-row:2}.banner-team .identity{font-size:22px}.banner-team .logo{width:48px;height:48px}}
/* .headline-table-light: the white/dark-text look of the "Copy" picture.
   The table itself isn't on the page -- picks_png() draws the picture from
   its HTML following these rules, so change both together. */
.headline-table-light{background:#fff;color:#222;border-collapse:collapse;font-size:12px;font-family:Graduate,Georgia,serif;width:auto}
.headline-table-light th,.headline-table-light td{padding:2px 6px;border:1px solid #ddd;text-align:center;line-height:1.15;white-space:nowrap}
.headline-table-light th{background:#f2f2f2;font-weight:600}
.headline-table-light img{height:16px;width:16px;object-fit:contain;vertical-align:middle}
main.headline-shell{overflow-x:auto}
.context-row{margin:12px 0}.context-value{font-size:12px;line-height:1.35;overflow-wrap:anywhere;color:#ccc}
.context-value:last-child{text-align:right}.matchup-head{font-size:15px}
@media(max-width:650px){.stat-line{grid-template-columns:minmax(90px,1.3fr) 65px minmax(75px,1fr) 65px;gap:5px;font-size:12px}.matchup-head{grid-template-columns:minmax(70px,1fr) minmax(50px,1.3fr) minmax(70px,1fr);gap:5px}.stat-name{font-size:11px}.stat-number .rank{display:block;margin:0}.matchup-head .side{font-size:11px}}
@media(max-width:600px){main.packet{padding:18px 16px}.stat-line{font-size:16px}.stat-name{font-size:14px}.packet h1{font-size:23px}.packet .metrics{gap:18px}.packet .metric strong{font-size:22px}}
@media print{body:has(.packet),.packet{background:white;color:#222}.packet .card{border-color:#aaa}.stat-impact.left,.stat-impact.right{color:#333}}
'''


STYLE += '''
/* Compact, portable report skin. Numeric cells retain a normal tabular font. */
/* NFL font on every title (h1/h2/h3, tab labels) and every team-abbreviation
   display (team-label, matchup-head's "NE offense" side labels) -- not on
   dense data-table headers/numbers, which stay legible in a plain font. */
.packet h1,.packet h2,.packet h3,.team-label,.matchup-head .side,.pkgtab-label{font-family:Graduate,Georgia,serif}
.packet h1{letter-spacing:.04em;font-weight:400}
.report-date{color:#aaa;margin:2px 0 10px;font-size:13px;font-family:Graduate,Georgia,serif}
.copy-picks{margin:0 0 14px;display:flex;align-items:center;gap:10px}
.copy-btn{font:12px Arial,sans-serif;background:#1a1d21;color:#e3e6e9;border:1px solid #41464e;border-radius:4px;padding:6px 14px;cursor:pointer}
.copy-btn:hover{background:#23272c}
/* Confirmation toast next to the button -- opacity 0 at rest; .show plays
   a fade-in/hold/fade-out over ~3s. Removing+re-adding .show (in the JS)
   needs a forced reflow in between or a second click before the first
   animation finishes won't restart it. */
.copy-feedback{font:12px Arial,sans-serif;color:#8bd9ab;opacity:0}
.copy-feedback.show{animation:copyFeedbackFade 3s ease forwards}
@keyframes copyFeedbackFade{0%{opacity:0}10%{opacity:1}70%{opacity:1}100%{opacity:0}}
/* Copy (see copy_picks_widget): the picture stays hidden unless the button
   falls back to showing it -- then it's the thing to right-click/long-press,
   scaled down to fit a phone, and the label reads "Done" to put it away. */
.copy-toggle,.copy-close,.copy-hint,.copy-png{display:none}
.copy-hint{font:12px Arial,sans-serif;color:#aeb8c3}
.copy-toggle:checked~.copy-picks .copy-open{display:none}
.copy-toggle:checked~.copy-picks .copy-close,.copy-toggle:checked~.copy-picks .copy-hint{display:inline}
.copy-toggle:checked~.copy-png{display:block;max-width:100%;height:auto;margin:0 0 14px}
.packet-bundle{align-content:flex-start;align-items:flex-start;box-sizing:border-box;padding:16px 20px}
.packet-bundle::after{content:'';order:1;flex:0 0 100%;height:0}
.packet-bundle .pkgtab-label,.packet-tabs a{padding:9px 18px;border:1px solid transparent;border-bottom:1px solid #393d42;border-radius:8px 8px 0 0;margin-bottom:0}
.packet-bundle .pkgtab-radio:checked+.pkgtab-label,.packet-tabs a[aria-current="page"]{background:#181b1f;color:#8fc9ef;border:1px solid #393d42;border-bottom-color:#181b1f}
.packet-tabs{justify-content:flex-start;gap:0;padding:0;position:static}
.packet-bundle .pkgtab-radio{display:block;position:absolute;opacity:0;width:1px;height:1px}
.pkgtab-radio:focus-visible+.pkgtab-label{outline:2px solid #8fc9ef;outline-offset:-3px}
main.pkgpanel{box-sizing:border-box;width:100%;margin:0;padding:20px 0}
#pt-stats:checked~.pkgpanel[data-tab=stats]{display:block}#pt-specs:checked~.pkgpanel[data-tab=specs]{display:block}
/* Specs tab (specs_page): label | value rows, one column on a phone. */
.spec-card h2{margin-top:0;font-size:19px}
.spec-card.tier-guide{font-size:13px}.spec-card.tier-guide summary{display:none}
.spec-prose{margin:0;font-size:13px;line-height:1.45;color:#c7ced5}.spec-list{display:grid;grid-template-columns:minmax(120px,200px) minmax(0,1fr);gap:7px 18px;margin:0;font-size:13px;line-height:1.45}
.spec-list dt{color:#9ba8b5}.spec-list dd{margin:0;overflow-wrap:anywhere}.spec-list .spec-list{grid-template-columns:minmax(110px,170px) minmax(0,1fr)}
.spec-list ul{margin:0;padding-left:18px}
@media(max-width:650px){.spec-list,.spec-list .spec-list{grid-template-columns:1fr;gap:2px}.spec-list dd{margin-bottom:8px}}
.stats-table{background:#111;color:#e3e6e9;font:12px/1.25 Arial,sans-serif;font-variant-numeric:tabular-nums;border-collapse:collapse}
/* The headline sheet is all NFL font, headers and figures included. */
.headline-table{background:#111;color:#e3e6e9;font:12px/1.15 Graduate,Georgia,serif;font-variant-numeric:tabular-nums;border-collapse:collapse}
/* The page and the picture are one sheet drawn twice, so the dark table
   states the same base its light twin does. Without this it inherited the
   stylesheet's global `td,th{text-align:right}`, which right-aligned the
   team codes, quarterbacks and picks on the page while the picture
   centred them -- the two never looked alike and nothing said why. */
.headline-table td,.headline-table th{text-align:center;line-height:1.15}
.headline-table th{font-weight:600}
.headline-table{width:auto}
/* Stats tables size to their own content (3-6 columns of team/value data)
   instead of stretching full-width. */
.stats-table{width:auto;max-width:100%}
/* CSS-only sorting: a <table> can't reorder its rows, a grid can. Rows and
   sections flatten into the grid (display:contents), every cell takes its
   row's --o as its `order`, and the headers stay on top at -1. Unsorted,
   --o is 0 everywhere and rows show in source (alphabetical) order.
   Column count comes from each table's inline grid-template-columns. */
table.stats-table{display:grid;width:max-content;max-width:none}
.stats-table thead,.stats-table tbody,.stats-table tr{display:contents}
.stats-table td{order:var(--o,0)}.stats-table th{order:-1}
/* Grid cells stretch instead of centering like table cells -- a 20px line
   (the logo's height) keeps text and logos lined up in every row. */
.packet .stats-table td{line-height:20px}.stats-table td img{vertical-align:top}
.stats-table tbody tr:hover td{background:#1d2126}
.packet .headline-table th{padding:2px 6px}
/* Confidence key, bottom right under the sheet (tier_legend). */
.model-credit{margin:10px 0 0;font:12px Arial,sans-serif;color:#b7c0c9;text-align:right}
.tier-legend{display:flex;flex-wrap:wrap;align-items:center;justify-content:flex-end;gap:4px 16px;margin:6px 0 0;font:11px Arial,sans-serif;color:#b7c0c9;text-align:right}
.tier-legend strong{font:12px Graduate,Georgia,serif;color:#e3e6e9}
.tier-key{display:flex;align-items:center;gap:5px;white-space:nowrap}
.tier-key i{display:inline-block;width:10px;height:10px;border-radius:2px;flex:0 0 auto}
/* The dash entry: an empty outline, because it marks the absence of a pick. */
.tier-swatch-none{border:1px solid #6b7480;background:transparent}
.tier-note{color:#8e99a5;flex-basis:100%;text-align:right}
.tier-gap{border-left:3px solid #6b7480;padding:6px 12px;margin:8px 0;font:12px Arial,sans-serif;color:#b7c0c9}
@media(max-width:650px){.tier-legend{align-items:flex-start;text-align:left}}
/* The expandable key under the sheet (tier_guide). */
.tier-guide{margin:10px 0 0;font:12px/1.5 Arial,sans-serif;color:#c8cdd3}
.tier-guide summary{cursor:pointer;font:13px Graduate,Georgia,serif;color:#e3e6e9;text-align:right}
.tier-guide h4{font:13px Graduate,Georgia,serif;color:#e3e6e9;margin:14px 0 4px}
.tier-qualify{margin:0 0 8px;color:#9ba8b5}
.tier-table{width:100%;border-collapse:collapse;font-size:12px}
.tier-table th,.tier-guide p,.tier-guide h4{text-align:left}
.tier-table th{text-align:left;font-weight:600;color:#9ba8b5;border-bottom:1px solid #41464e;padding:4px 8px 4px 0}
.tier-table td{vertical-align:top;padding:5px 8px 5px 0;border-bottom:1px solid #23272c;text-align:left}
.tier-table .tier-record{color:#b7c0c9;white-space:nowrap}
.tier-note-row td{color:#8e99a5;border-bottom:1px solid #23272c;padding-top:0}
.tier-chip{display:inline-block;width:20px;text-align:center;border-radius:3px;color:#222;font-weight:700}
.tier-chip-none{border:1px dashed #6b7480;color:#9aa4ae;font-weight:400}
.tier-guide .tier-note{color:#8e99a5;margin:12px 0 0}
@media(max-width:650px){.tier-table th:nth-child(6),.tier-table td:nth-child(6),
.tier-table th:nth-child(7),.tier-table td:nth-child(7){display:none}.tier-guide summary{text-align:left}}
.tier-warn{color:#ffd48a}
.packet .headline-table td{padding:2px 6px}
/* Every stats table formatted the same tight way (this used to be QB
   Elo-only, leaving the others visibly looser/wider) -- small logos,
   snug padding, all of them. */
.packet .stats-table th,.packet .stats-table td{padding:2px 6px}
.packet .headline-table th,.packet .stats-table th{background:#1a1d21;color:#aeb8c3;border:0;border-bottom:2px solid #41464e;white-space:nowrap}
.packet .headline-table td,.packet .stats-table td{border:0;border-bottom:1px solid #292d32;white-space:nowrap}
.headline-table tbody tr:hover,.stats-table tbody tr:hover{background:#1d2126}
.headline-table img{height:16px;width:16px;object-fit:contain;vertical-align:middle}
.stats-table img{height:20px;width:22px;object-fit:contain;vertical-align:middle}
/* The quarterbacks are reference, not numbers to compare -- quiet, a size
   down, with the rating tucked against the name it belongs to.
   Positions come from HEADLINE_COLUMNS (6 and 10 are the names, 7 and 11
   the ratings). These were written for an older column order and, when
   the sheet gained its kickoff and rating columns, kept styling whatever
   had moved into positions 4 and 7 -- which set the away rating in Arial
   while the home rating stayed in the NFL font. `font:` shorthand is
   avoided here for that reason: it silently resets the family. */
.stats-table .rank{font-size:10px;line-height:1;color:#aab2bc;margin-left:6px}
.stats-table td:first-child,.stats-table th:first-child{text-align:left}
.sort-radio{display:none}
.stats-table th label{font-weight:600;cursor:pointer}
.stats-table .sort-to-d{display:none}.stats-table .sort-to-d:after{content:' ↑'}
/* width:fit-content -- otherwise this (a plain block div) stretches to
   fill its .stats-grid column, which is sized to the WIDEST table sharing
   that column (e.g. Rushing sharing QB Elo's column, Misc sharing
   Passing's) -- leaving a gap of empty space between a narrower table's
   own content and its scrollbar, parked out at the wide column's edge. */
.table-scroll{overflow-x:auto;width:fit-content;max-width:100%}
.banner-team .logo{width:70px;height:70px}
/* Every stats table lists all 32 teams -- scrolls after ~10 instead of
   pushing the rest of the section down the page; sticky header keeps the
   column labels in view while scrolling. */
.table-scroll.scroll-tall{max-height:360px;overflow-y:auto;margin-bottom:16px}
.table-scroll.scroll-tall thead th{position:sticky;top:0;z-index:1}
/* A single-metric table (QB Elo, currently the only one) leads with a
   Rank column (plain "1", not the inline (#1) badge multi-metric tables
   use) -- override the generic first-child-is-Team left-align just for
   those: Rank centered, Team (now 2nd) left. Class-based, not #id-based,
   since the Raw/Model toggle below renders two copies of this table with
   different ids (qb-elo-raw/qb-elo-model). */
.stats-table-ranked td:first-child,.stats-table-ranked th:first-child{text-align:center}
.stats-table-ranked td:nth-child(2),.stats-table-ranked th:nth-child(2){text-align:left}
/* One table per row -- Passing, then Rushing below it, then Misc -- at
   every width, so a phone never gets two tables squeezed side by side.
   minmax(0,1fr), not 1fr: a bare 1fr track can't shrink below its widest
   table, which forced the whole column ~880px wide on a 390px phone and
   clipped every table's right edge; this keeps the column at screen width
   so a wide table scrolls inside its own .table-scroll box instead. */
.stats-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:4px 28px;align-items:start}
/* Raw/Model toggle -- CSS-only :checked~sibling trick, same idea as the
   packet bundle's own tabs; no JS round-trip since this is a static file. */
.stats-mode-radio{display:none}
.stats-mode-label{display:inline-block;font:12px Arial,sans-serif;padding:6px 14px;border:1px solid #41464e;color:#aeb8c3;cursor:pointer}
.stats-mode-label:first-of-type{border-radius:4px 0 0 4px}
.stats-mode-label:last-of-type{border-radius:0 4px 4px 0;border-left:0}
.stats-mode-radio:checked+.stats-mode-label{background:#1a1d21;color:#fff}
.stats-view{display:none}
#stats-mode-raw:checked~#stats-view-raw{display:block}
#stats-mode-model:checked~#stats-view-model{display:block}
.banner-team .team-copy{display:flex;flex-direction:column}.banner-qb{font:11px/1.4 Arial,sans-serif;color:#aaa;white-space:normal;margin-top:4px}
.packet .pill.pick{background:#cfd5db;color:#222}.pill.pick.tier-S{background:#e3c4ff}.pill.pick.tier-A{background:#b9e4c4}.sheet-notes{border-top:1px solid #333;margin-top:28px;padding-top:12px}.sheet-notes summary{cursor:pointer;font:15px Graduate,Georgia,serif}
.sheet-notes p{font-size:13px;line-height:1.5;color:#c8cdd3;margin:10px 0}.sheet-notes strong{color:#eee}
.pick-header{overflow-x:auto;border-bottom:1px solid #333;padding:10px 0 16px;margin-bottom:12px}
.pick-grid{display:grid;grid-template-columns:minmax(120px,1fr) auto minmax(120px,1fr);align-items:center;gap:4px 10px;text-align:center;font-variant-numeric:tabular-nums}
/* The five numbers stay one tight group between the team names rather than
   being stretched over the full width, and each keeps its label on top. */
.pick-numbers{display:flex;justify-content:center;align-items:flex-end;gap:14px}
.pick-cell{display:flex;flex-direction:column;align-items:center;gap:1px;min-width:46px}
.pick-label{font:9px Arial,sans-serif;letter-spacing:.03em;text-transform:uppercase;color:#8c99a6}
.pick-qb{text-align:left}.pick-qb.home{text-align:right}
.record{font:13px Arial,sans-serif;color:#9ba8b5;font-weight:400}
@media(max-width:650px){.pick-grid{gap:2px 6px}.pick-numbers{gap:8px}.pick-cell{min-width:38px}
 .pick-value{font-size:14px}.pick-team{font-size:18px;gap:4px}.pick-team .logo{width:38px;height:38px}
 .record{font-size:11px}.pick-qb{font-size:11px}}
.pick-team{display:flex;flex-direction:column;align-items:flex-start;gap:1px;font:24px Graduate,serif}
.pick-team.home{align-items:flex-end}
.pick-team .identity{display:flex;flex-wrap:wrap;align-items:center;gap:6px}
.pick-team .logo{width:52px;height:52px}
.pick-value{font-size:15px;font-weight:600}.pick-qb{font:12px Graduate,Georgia,serif;white-space:nowrap}
.pick-score{grid-column:2;display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:baseline;font:12px Graduate,Georgia,serif;color:#9ba8b5}
.pick-score .score-label{justify-self:end}.pick-score .score-value{color:#eee}
summary.matchup-summary{padding-left:0}summary.matchup-summary:before{left:-12px}
.site-row{display:grid;grid-template-columns:var(--gutter) minmax(0,1fr) var(--gutter);gap:var(--row-gap);align-items:center;margin:12px 0;font-size:11px}
.site-label{white-space:nowrap}.site-label strong{margin-right:8px}.site-row .context-value{font-size:11px}
.site-row .context-value{text-align:right}
/* Reference facts under the context bars -- separated by dots, wrapping to a
   second line on a phone rather than being cut off. */
.game-facts{display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 0;margin:10px 0 2px;font:11px Arial,sans-serif;color:#9ba8b5}
.game-facts span:not(:last-child):after{content:'·';margin:0 7px;color:#5d6873}
/* Weather: a heading, then one tight row per model input (matchup_attribution). */
.context-group{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-size:11px;font-weight:700;margin:14px 0 2px}
.context-group .context-value{font-size:11px;font-weight:400}
.site-row.weather-factor{margin:5px 0}.weather-factor .site-label{padding-left:12px;color:#ccc}
/* Closing line: baseline + every contribution + residual = the header's number. */
@media(max-width:650px){.packet{--gutter:130px;--val-w:55px;--row-gap:5px}.stat-line{grid-template-columns:calc(var(--gutter) - var(--val-w) - var(--row-gap)) var(--val-w) minmax(0,1fr) var(--gutter)}.matchup-head{grid-template-columns:var(--gutter) minmax(0,1fr) var(--gutter)}.site-label{white-space:normal}}
@media(max-width:650px){.packet-bundle{padding:10px}.packet-bundle .pkgtab-label{padding:8px;font-size:12px}}
@media print{main.pkgpanel{display:block!important}.pkgtab-label,.pkgtab-radio{display:none!important}}
'''


@lru_cache(maxsize=1)
def team_colors():
    path = Path('data/logos/team_colors.json')
    return json.loads(path.read_text()) if path.exists() else {}


def team_color(team):
    aliases = {'LA': 'LAR', 'STL': 'LAR', 'SD': 'LAC', 'OAK': 'LV'}
    color = team_colors().get(aliases.get(team, team), '#888888')
    return color if len(color) == 7 and color[0] == '#' and all(c in '0123456789abcdefABCDEF' for c in color[1:]) else '#888888'


def lookback_span(season, week, lookback):
    """Series indexed by (season, week), True where that week had a
    regular-season game -- covers the `lookback` REGULAR-season weeks
    counted backward from the target week, but keeps the whole window: any
    playoff weeks caught inside that span (a lookback reaching back across
    a season boundary) are included in the returned span too -- they just
    don't themselves consume one of the `lookback` backward steps, since
    they're not in `regular` to begin with."""
    sched = pd.read_parquet('data/sched.parquet', columns=['season', 'week', 'game_type'])
    prior = sched[(sched.season < season) | ((sched.season == season) & (sched.week < week))]
    weeks = prior.groupby(['season', 'week']).game_type.apply(lambda x: x.eq('REG').any()).sort_index()
    regular = weeks[weeks].index
    if not len(regular):
        return weeks.iloc[0:0]
    start = regular[-min(lookback, len(regular))]
    return weeks.loc[start:]


def lookback_description(season, week, lookback):
    """Plain-English methodology note for the Stats tab -- what the
    rolling window actually covers (with the real week range) and what QB
    Elo means, since "(#1) is best" alone doesn't say much."""
    span = lookback_span(season, week, lookback)
    if span.empty:
        return ''
    (first_season, first_week), (last_season, last_week) = span.index[0], span.index[-1]
    span_text = (f'{int(first_season)} Week {int(first_week)} through {int(last_season)} Week {int(last_week)}'
                if first_season != last_season else f'{int(first_season)} Weeks {int(first_week)}–{int(last_week)}')
    playoff_weeks = int((~span).sum())
    playoff_note = (f' {playoff_weeks} of those are playoff weeks -- included in the numbers below, but they '
                    f'don\'t themselves count as one of the {lookback}.' if playoff_weeks else '')
    return ('<details><summary>How these stats are calculated</summary>'
           f'<p>A {lookback}-week rolling lookback of regular-season games, counted backward from this week '
           f'({span_text}).{playoff_note} Every rate here is an unweighted, observed average over that window -- '
           'not a model input (the model itself uses a separately recency-weighted, standardized version of these '
           'same underlying stats). QB Elo is a recency-weighted rating of that quarterback\'s own game production '
           '(not the team\'s win/loss record); under Defense it\'s the same rating averaged over the opposing '
           'QBs actually faced, i.e. strength of the passing competition seen so far. (#1) is best; click a '
           'column header to sort by rank.</p>'
           '<p>Raw: flat average over the lookback window. Model: the same plays, weighted the way the model '
           'itself weighs them (recent games count more). QB Elo is already recency-weighted either way, '
           'so it doesn\'t change between the two.</p></details>')


def display_stats(season, week, lookback, calculation='mean'):
    """Observed rates, strictly pregame; never model inputs (the model
    always uses its own separately-standardized version of these, even
    when calculation matches). calculation='mean' (default): unweighted
    average over the lookback window -- "raw". calculation='steep': the
    same recency decay the model itself uses for its own
    features -- "model". Same plays, same window, different pooling."""
    import data_crunchski_2 as dc
    import utils
    sources = [__file__, 'data_crunchski_2.py', 'data/sched.parquet']
    sources += list(Path('data/pbp').glob('pbp_*.parquet'))
    cached = utils.cache_path('packet_stats', [int(season), int(week), lookback, calculation], sources)
    if cached.exists():
        return pd.read_parquet(cached)
    sched = pd.read_parquet('data/sched.parquet').replace(
        {'away_team': dc.RELOCATED_TEAMS, 'home_team': dc.RELOCATED_TEAMS})
    span = lookback_span(season, week, lookback)
    if span.empty:
        return pd.DataFrame()
    selected = span.index
    frames = []
    for year in selected.get_level_values(0).unique():
        path = Path(f'data/pbp/pbp_{year}.parquet')
        if not path.exists():
            return pd.DataFrame()  # Do not present partial-history league ranks.
        data = pd.read_parquet(path)
        frames.append(data[data.week.isin([w for s, w in selected if s == year])])
    plays = pd.concat(frames)
    # calc_stats reads RATE_MODE as a module-level global rather than a
    # parameter -- set it explicitly (not just trust whatever an earlier,
    # unrelated call in this same process left it as) and put it back
    # afterward so this "just displaying stats" call doesn't leak state
    # into whatever runs next.
    previous_rate_mode, previous_leverage = dc.RATE_MODE, dc.GAME_IMPORTANCE
    dc.RATE_MODE = calculation
    # An importance-weighted preset needs the leverage table too, or it would
    # quietly fall back to plain recency weighting (see _pool_ingredients).
    if calculation in dc.IMPORTANCE_WEIGHT:
        dc.GAME_IMPORTANCE = dc.game_leverage(sorted(selected.get_level_values(0).unique()), (int(season), int(week)))
    # calc_stats needs game_date to compute recency decay for anything
    # other than a flat 'mean' -- only safe to drop it in the 'mean' case.
    plays_in = plays.drop(columns='game_date', errors='ignore') if calculation == 'mean' else plays
    try:
        result = dc.calc_stats(plays_in).reset_index()
    finally:
        dc.RATE_MODE, dc.GAME_IMPORTANCE = previous_rate_mode, previous_leverage
    qb, defense = dc.calc_qb_elo(plays, sched)
    # Current scheduled starters; most recent scheduled starter for teams on bye.
    known = sched[(sched.season < season) | ((sched.season == season) & (sched.week <= week))]
    id_column = lambda side: [f'{side}_qb_id'] if f'{side}_qb_id' in known else []
    starters = pd.concat([known[['season', 'week', f'{side}_team', f'{side}_qb_name'] + id_column(side)].rename(
        columns={f'{side}_team': 'team', f'{side}_qb_name': 'name', f'{side}_qb_id': 'id'})
        for side in ['away', 'home']])
    starters = starters.dropna(subset=['name']).sort_values(['season', 'week']).drop_duplicates('team', keep='last')
    starters['team'] = starters.team.replace(dc.RELOCATED_TEAMS)
    starters['off_qb_elo'] = dc.starter_qb_elo(starters, qb)
    ratings = starters[['team', 'off_qb_elo']]
    result = result.merge(ratings, on='team', how='left', validate='one_to_one').merge(
        defense, on='team', how='left', validate='one_to_one')
    utils.save_parquet(result, cached)
    return result


# Per-play metrics measured in points rather than as a share of plays.
POINTS_PER_PLAY = {'pass_epa_pp', 'run_epa_pp'}


def stat_cell(stats, team, unit, metric, rank_before=False):
    column = f'{unit}_{metric}'
    if stats.empty or column not in stats or team not in stats.team.values:
        return '—'
    values = stats.set_index('team')[column].replace([np.inf, -np.inf], np.nan)
    value = values.loc[team]
    if pd.isna(value):
        return '—'
    # '_pp' means per play, which is a RATE for first downs, turnovers and
    # penalties (33.3% of plays) but POINTS for EPA (0.153 points a play).
    # Formatting the second as the first turned expected points added into a
    # percentage of nothing.
    formatted = (f'{value:+.3f}' if metric in POINTS_PER_PLAY
                 else f'{value:.1%}' if '%' in metric or metric.endswith('_pp')
                 else f'{value:.1f}')
    lower_off = metric in ['turnovers_pp', 'stuff_%', 'sack_%', 'qb_hit_%', 'penalties_pp']
    ascending = lower_off if unit == 'off' else not lower_off
    if metric == 'penalties_pp':
        ascending = True  # Fewer flags, for either possession-based unit.
    if metric == 'qb_elo':
        ascending = unit == 'def'  # Offense higher first; defense lower first.
    rank = values.rank(method='min', ascending=ascending).loc[team]
    badge = f'<span class="rank">(#{int(rank)})</span>'
    return badge + ' ' + formatted if rank_before else formatted + badge


def point_bar(value, scale, row, left_team=None):
    if row.get('market') == 'total':
        width = 44 * abs(value) / max(scale, .01)
        edge = 50 + width if value >= 0 else 50 - width
        start = 50 if value >= 0 else edge
        color = '#3d9275' if value >= 0 else '#b67f48'
        anchor = 'none' if value >= 0 else 'translateX(-100%)'
        signed = value if abs(value) >= .05 else 0
        return (f'<div class="contribution"><span class="bar" style="background:{color};left:{start:.2f}%;width:{width:.2f}%"></span>'
                f'<b style="position:absolute;left:{edge:.2f}%;transform:{anchor};padding:0 3px">{signed:+.1f}</b></div>')
    if row.get('market') == 'spread':
        width = 44 * abs(value) / max(scale, .01)
        favored = row.away_team if value > 0 else row.home_team
        toward_left = favored == (left_team or row.away_team)
        edge = 50 - width if toward_left else 50 + width
        start = 50 - width if toward_left else 50
        signed = -value if abs(value) >= .05 else 0
        anchor = 'translateX(-100%)' if toward_left else 'none'
        return (f'<div class="contribution"><span class="bar" style="background:{team_color(favored)};left:{start:.2f}%;width:{width:.2f}%"></span>'
                f'<b style="position:absolute;left:{edge:.2f}%;transform:{anchor};padding:0 3px">{signed:+.1f}</b></div>')
    if row.get('market') == 'spread':
        label = row.away_team if value > 0 else row.home_team
    else:
        label = 'Over' if value > 0 else 'Under'
    label = f'{label} {abs(value):.2f}' if value else '0.00'
    width = 49 * abs(value) / max(scale, .01)
    left = 50 if value >= 0 else 50 - width
    if left_team is not None:
        favored = row.away_team if value > 0 else row.home_team
        left = 50 - width if favored == left_team else 50
    color = f'background:{team_color(row.away_team if value > 0 else row.home_team)};' if row.get('market') == 'spread' else ''
    return (f'<div class="contribution"><span class="bar {"negative" if value < 0 else ""}" '
            f'style="{color}left:{left:.2f}%;width:{width:.2f}%"></span><b>{escape(label)} pts</b></div>')


def comparison_bars(stats, offense, defense, metric):
    """Bar lengths compare observed rates, not ranks or model attribution."""
    if metric == 'qb_elo' or stats.empty:
        return ''
    data = stats.set_index('team')
    left = data.loc[offense].get(f'off_{metric}', np.nan) if offense in data.index else np.nan
    right = data.loc[defense].get(f'def_{metric}', np.nan) if defense in data.index else np.nan
    if not np.isfinite([left, right]).all() or min(left, right) < 0:
        return ''
    share = left / (left + right) if left + right else .5
    return (f'<div class="stat-bars" aria-label="Observed rate comparison">'
            f'<span class="left" style="flex:{share:.6f}"></span>'
            f'<span class="right" style="flex:{1-share:.6f}"></span></div>')


@lru_cache(maxsize=1)
def packet_schedule():
    import data_crunchski_2 as dc
    import travel
    sched = pd.read_parquet('data/sched.parquet').replace(
        {'away_team': dc.RELOCATED_TEAMS, 'home_team': dc.RELOCATED_TEAMS})
    # Travel miles per team (travel.py, from the same stadium coordinates the
    # weather pull uses) -- shown next to the Travel bar whether or not the
    # running model version takes them as an input.
    return sched.merge(travel.game_travel(sched), on='game_id', how='left', validate='one_to_one')


# data/sched.parquet's surface codes, as they're written on the page.
SURFACES = {'grass': 'Grass', 'fieldturf': 'FieldTurf', 'matrixturf': 'MatrixTurf', 'sportturf': 'SportTurf',
            'astroturf': 'AstroTurf', 'a_turf': 'A-Turf'}


def schedule_game(row):
    """This game's data/sched.parquet row (stadium, roof, surface, rest,
    referee...), or None if it isn't there."""
    if 'season' not in row:
        return None
    sched = packet_schedule()
    match = sched[(sched.season == row.season) & (sched.week == row.week) &
                  (sched.away_team == row.away_team) & (sched.home_team == row.home_team)]
    return None if match.empty else match.iloc[0]


def context_cells(feature, row):
    label = pretty(feature)
    if feature == 'context_importance':
        cells = [f'{row[side + "_importance"]:.0%}' if pd.notna(row.get(side + '_importance')) else '—'
                 for side in ['away', 'home']]
        return 'Playoff leverage' if 'importance_method' in row else 'Game importance', cells[0], cells[1]
    if feature == 'context_weather':
        # An older model's single combined weather feature: just the bar.
        return 'Weather', '', ''
    if feature not in ['away_rest_adv', 'away_travel_adv', 'home_field_adv', 'context_referee']:
        return label, '', ''
    game = schedule_game(row)
    if game is None:
        return label, '', ''
    if feature == 'context_referee':
        name = game.get('referee')
        name = str(name) if pd.notna(name) else 'Unassigned'
        average, count = row.get('referee_avg_total'), row.get('referee_prior_games')
        detail = f'Avg {average:.1f} · n={int(count)}' if pd.notna(average) and pd.notna(count) else ''
        return 'Referee', name, detail
    if feature == 'home_field_adv':
        # Stadium and surface on the right of the bar. 'location' is a Home/Neutral
        # site-type flag, not a city -- only worth adding for a neutral site.
        venue = str(game.get('stadium')) if pd.notna(game.get('stadium')) else ''
        surface = str(game.get('surface')).strip() if pd.notna(game.get('surface')) else ''
        neutral = str(game.get('location', '')).lower() == 'neutral'
        return label, '', ' · '.join(p for p in [venue, SURFACES.get(surface.lower(), surface.title()),
                                                'neutral site' if neutral else ''] if p)
    if feature == 'away_travel_adv':
        away, home = game.get('away_travel_miles'), game.get('home_travel_miles')
        if pd.isna(away) or pd.isna(home):
            return 'Travel', '', ''
        return 'Travel', f'{row.away_team} {away:,.0f} mi', f'{row.home_team} {home:,.0f} mi'
    away, home = game.get('away_rest'), game.get('home_rest')
    if pd.isna(away) or pd.isna(home):
        return label, '', ''
    return 'Rest', f'{row.away_team} {away:g}d', f'{row.home_team} {home:g}d'


def game_facts(row):
    """The reference line under the context bars: when, where, on what, in
    what, and who is officiating. One line, two at most on a phone. None of
    it is a model input except the weather numbers, which appear as their own
    bars above -- this is the context a human wants when reading a pick."""
    game = schedule_game(row)
    if game is None:
        return ''
    kickoff, readings = weather_details(row)
    surface = str(game.get('surface')).strip() if pd.notna(game.get('surface')) else ''
    venue = str(game.get('stadium')) if pd.notna(game.get('stadium')) else ''
    if str(game.get('location', '')).lower() == 'neutral':
        venue += ' · neutral site'
    def reading(key):
        text = re.sub(r'\s*[(][^)]*[)]', '', readings.get(key, '')).strip()
        return '' if key == 'precip_inches' and text.startswith(('0.00', '0 ')) else text

    wet = reading('precip_inches')
    conditions = ' · '.join(p for p in [reading('feels_like_f'), reading('wind_mph'), wet or 'dry'] if p)
    referee = str(game.get('referee')) if pd.notna(game.get('referee')) else ''
    travel = game.get('away_travel_miles')
    trip = f'{row.away_team} travelled {travel:,.0f} mi' if pd.notna(travel) and travel >= 1 else ''
    parts = [kickoff, venue, SURFACES.get(surface.lower(), surface.title()), conditions, trip,
             f'Ref: {referee}' if referee else '']
    shown = ''.join(f'<span>{escape(part)}</span>' for part in parts if part)
    return f'<div class="game-facts">{shown}</div>' if shown else ''


def matchup_attribution(row, stats, panel, shared=False, differential=False):
    direction = 1 if row.get('market') == 'total' else -1
    # One entry per model feature, never combined or dropped for display:
    # offense/defense metrics go in the two expandable matchup bars, every
    # other feature gets its own row below them.
    values = {c[5:]: float(row[c]) for c in row.index if c.startswith('attr_') and pd.notna(row[c])}
    scale = max([abs(v) for v in values.values()] + [.01])
    net_scale = max([abs(sum(v for f, v in values.items() if f.startswith(prefix)))
                     for prefix in ['away_off_', 'away_def_']] + [.01])
    sections, used = [], set()
    order = ['qb_elo', 'pass_ypp', 'pass_completion_%', 'explosive_pass_%', 'sack_%', 'qb_hit_%',
             'run_ypp', 'explosive_run_%', 'stuff_%', 'first_down_pp', 'series_success_%',
             'third_down_%', 'fourth_down_%', 'turnovers_pp', 'penalties_pp']
    for prefix, offense, defense in [('away_off_', row.away_team, row.home_team),
                                      ('away_def_', row.home_team, row.away_team)]:
        features = sorted([f for f in values if f.startswith(prefix)],
                          key=lambda f: (order.index(f[len(prefix):]) if f[len(prefix):] in order else len(order), f))
        if not features:
            continue
        rows = []
        away_unit, home_unit = ('off', 'def') if prefix == 'away_off_' else ('def', 'off')
        for feature in features:
            metric = feature[len(prefix):]
            cells = [stat_cell(stats, row.away_team, away_unit, metric),
                     stat_cell(stats, row.home_team, home_unit, metric, rank_before=True)]
            label = pretty(metric)
            value = values[feature]
            rows.append(f'<div class="stat-row"><div class="stat-line"><div class="stat-name">{escape(label)}</div><div class="stat-number">{cells[0]}</div>'
                        f'{point_bar(value, scale, row, left_team=row.away_team)}'
                        f'<div class="stat-number">{cells[1]}</div></div></div>')
            used.add(feature)
        net = direction * sum(values[f] for f in features)
        left_label = 'offense' if away_unit == 'off' else 'defense'
        right_label = 'offense' if home_unit == 'off' else 'defense'

        # <details> gives a native collapse/expand triangle for free -- closed
        # by default (no `open` attribute), so the page stays compact until
        # someone actually wants the per-metric breakdown. The grid/flex
        # layout lives on a plain nested div, not on <summary> itself --
        # see the CSS comment for why.
        # Logos bookend the row -- away logo on the far left, home logo on
        # the far right -- so text order flips per side: logo-then-text on
        # the left, text-then-logo on the right.
        sections.append(f'<details class="matchup"><summary class="matchup-summary"><div class="matchup-head">'
                        f'<div class="side">{logo(row.away_team)}{escape(row.away_team)} {left_label}</div>'
                        f'<div class="net-bar" aria-label="Net matchup contribution">{point_bar(direction * net, net_scale, row)}</div>'
                        f'<div class="side">{escape(row.home_team)} {right_label}{logo(row.home_team)}</div></div></summary>{"".join(rows)}</details>')
    other = []
    context_order = ['home_field_adv', 'context_stadium', 'context_field', 'away_rest_adv', 'away_travel_adv',
                     'context_referee',
                     'context_importance', 'context_weather']
    context_features = sorted((f for f in values if f not in used and not f.startswith('context_weather_')),
                              key=lambda f: (context_order.index(f) if f in context_order else len(context_order), f))
    for feature in context_features:
        value = values[feature]
        label, left, right = map(escape, context_cells(feature, row))
        if feature == 'home_field_adv':
            other.append(f'<div class="site-row"><div class="site-label">{label}{left}</div>'
                         f'{point_bar(value, scale, row)}<div class="context-value">{right}</div></div>')
            continue
        other.append(f'<div class="stat-line context-row"><div class="stat-name">{label}</div>'
                     f'<div class="context-value">{left}</div>{point_bar(value, scale, row)}'
                     f'<div class="context-value">{right}</div></div>')
    other.append(game_facts(row))
    # two_sided_packet's weather inputs, one row each under a small heading
    # carrying the kickoff time, each with the value it had on the right.
    weather_order = ['feels_like_f', 'wind_mph', 'precip_inches']
    weather = sorted((f for f in values if f.startswith('context_weather_') and f not in used),
                     key=lambda f: (weather_order.index(f[16:]) if f[16:] in weather_order else len(weather_order), f))
    if weather:
        kickoff, readings = weather_details(row)
        other.append(f'<div class="context-group"><span>Weather</span><span class="context-value">{escape(kickoff)}</span></div>' + ''.join(
            f'<div class="site-row weather-factor"><div class="site-label">{escape(re.sub(r" [(].*[)]$", "", pretty(f)))}</div>'
            f'{point_bar(values[f], scale, row)}<div class="context-value">{escape(readings.get(f[16:], ""))}</div></div>'
            for f in weather))
    total = sum(values.values())
    residual = row.prediction - row.baseline - total
    input_note = ('The shared model uses raw role-specific inputs with training-only standardization. Each displayed metric sums offense and opposing-defense effects. '
                  if shared else 'Display rates are separate from the model’s recency-weighted, league-ranked inputs. ')
    if differential:
        input_note = ('Each model input is raw offense minus opposing-defense stat, standardized using prior training data. '
                      'Displayed team rates are unweighted; model rate calculations retain the legacy recency weighting. ')
    if row.get('model_family') == 'joint':
        input_note += 'Both matchups and context interact through shared weights; margin and total are independently trained targets. '
    sign_note = ('Positive raises the total; negative lowers it. ' if direction == 1 else
                 'Negative contributions favor the away team; positive favor the home team, matching the away-team spread above. ')
    return ''.join(sections + other) + (
        f'<details><summary>Calculation notes</summary>Observed, unweighted rates over the pregame feature window; #1 is best, ties share rank. '
        f'Ranks include teams on bye, using their most recent scheduled QB when needed. QB ranks: offense higher first, defense lower first. Penalty ranks: fewer possession-based flags first, not necessarily flags committed by that unit. '
        f'Defensive QB Elo measures opposing QB game production allowed, lower is better. New feature builds include relief-QB production without league-average subtraction; older cached runs retain the previous centered metric. '
        f'{input_note}'
        f'{sign_note}'
        f'Section nets sum feature contributions, not predicted team scores. Net bars share a scale with each other; feature bars share their own scale. Display rounding can affect visible sums; calculations retain full precision. '
        f'Every model feature is shown separately; none are combined or left out. Referee average totals are shrunk toward the earlier league mean with 20 prior games of weight; same-week and future results are excluded. '
        f'Contributions explain the fitted prediction, not causal effects. Numerical residual {direction * residual:+.4f} points.</details>')


def page(title, content, theme=''):
    return f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{escape(title)}</title>{font_license()}<style>{font_style()}{STYLE}</style><main class="{escape(theme)}">{content}</main></html>'


@lru_cache(maxsize=1)
def font_style():
    path = Path(__file__).parent / 'assets/fonts/Graduate-Regular.ttf'
    encoded = base64.b64encode(path.read_bytes()).decode()
    return "@font-face{font-family:Graduate;src:url(data:font/ttf;base64," + encoded + ") format('truetype');font-display:swap}"


def font_license():
    license_text = (Path(__file__).parent / 'assets/fonts/OFL.txt').read_text()
    return '<!-- ' + license_text.replace('--', '—') + ' -->'


# Groups METRICS into digestible sub-tables instead of one wide Offense/
# Defense table -- Passing/Rushing/Misc for each unit; QB Elo (offense only,
# no defense has its own QB) gets its own single-metric table, built separately.
STAT_GROUPS = [
    ('Passing', ['pass_ypp', 'pass_epa_pp', 'pass_success_%', 'pass_completion_%', 'explosive_pass_%', 'sack_%', 'qb_hit_%']),
    ('Rushing', ['run_ypp', 'run_epa_pp', 'run_success_%', 'explosive_run_%', 'stuff_%']),
    ('Misc', ['series_success_%', 'first_down_pp', 'third_down_%', 'fourth_down_%', 'turnovers_pp', 'penalties_pp']),
]


def stats_tables(stats, games, season=None, week=None, lookback=None, weighted_stats=None):
    """League snapshot display ranks; no new fits or alternative model inputs.
    season/week/lookback (all three, or none) drive the "How these stats are
    calculated" methodology dropdown -- optional since some callers/tests
    only have the stats/games frames, not a real week to look up against
    data/sched.parquet. weighted_stats: same shape as `stats` but computed
    with the model's own recency decay (display_stats(..., calculation=
    'steep')) instead of a flat average -- when given, adds a Raw/Model
    toggle (CSS-only, no JS) instead of showing only the raw view. Column
    sorting is CSS-only too (see metric_table/sort_css)."""
    if stats.empty:
        return '<p class="muted">No pregame league snapshot available.</p>'
    names = {}
    sort_columns = {}  # table key -> column count, filled in by metric_table
    for side in ['away', 'home']:
        for _, row in games.iterrows():
            name = row.get(f'{side}_qb_name', row.get(f'{side}_qb_short'))
            if pd.notna(name):
                names[row[f'{side}_team']] = str(name)

    def metric_table(stats, unit, metrics, table_id, quarterback_column=False):
        metrics = [m for m in metrics if f'{unit}_{m}' in stats]
        if not metrics:
            return ''
        single = len(metrics) == 1
        sort_column = f'{unit}_{metrics[0]}'
        ordered = stats.sort_values(sort_column, ascending=False, na_position='last') if single and sort_column in stats else stats
        # Rank as its own leading column (plain "1", not "(#1)") only makes
        # sense for a single-metric table like QB Elo -- a multi-metric
        # table has no one overall rank, each metric keeps its own inline badge.
        heads = (['Rank'] if single else []) + ['Team'] + (['Quarterback'] if quarterback_column else []) + [pretty(m) for m in metrics]
        # Each cell's sort key rides along with it: a rank (missing = 999,
        # sorts last) for stat columns, the text itself for Team/Quarterback.
        rows = []
        for team in ordered.team:
            cells, keys = [], []
            if single:
                cell = stat_cell(stats, team, unit, metrics[0])
                rank = re.search(r'\(#(\d+)\)', cell)
                rank_number = rank.group(1) if rank else None
                cells.append(f'<td>{rank_number or "—"}</td>')
                keys.append(int(rank_number or 999))
            cells.append(f'<td class="team-label">{logo(team)} {escape(team)}</td>')
            keys.append(team.casefold())
            if quarterback_column:
                name = names.get(team, '—')
                cells.append(f'<td>{escape(name)}</td>')
                keys.append(name.casefold())
            for metric in metrics:
                cell = stat_cell(stats, team, unit, metric)
                rank = re.search(r'\(#(\d+)\)', cell)
                if single:
                    # Already broken out into its own Rank column above --
                    # no point repeating the same (#N) badge inline too.
                    cell = re.sub(r'\s*<span class="rank">.*?</span>', '', cell)
                cells.append(f'<td>{cell}</td>')
                keys.append(int(rank.group(1)) if rank else 999)
            rows.append((cells, keys))
        # Sorting is pure CSS so it works where scripts don't run (phone
        # file previews): every row carries its position under each sort
        # (--a3 = 3rd column best-first, --d3 = reversed) and a checked
        # radio (see sort_css below) copies one of them into the row's
        # grid `order`. Stable sorts, so ties keep alphabetical team order.
        positions = [[] for _ in rows]
        for i in range(len(heads)):
            for direction, reverse in [('a', False), ('d', True)]:
                ranked = sorted(range(len(rows)), key=lambda r: rows[r][1][i], reverse=reverse)
                for position, r in enumerate(ranked, 1):
                    positions[r].append(f'--{direction}{i + 1}:{position}')
        # Raw and Model render the same table twice (ids off-passing-raw /
        # off-passing-model) -- they share one set of radios, keyed without
        # the suffix, so flipping the toggle keeps whatever sort you picked.
        key = re.sub(r'-(raw|model)$', '', table_id)
        sort_columns[key] = len(heads)
        # Every stats table scrolls after ~10 rows instead of stretching the
        # page -- 32 teams is a lot of scrolling either way, so keep it
        # consistent across all of them, not just QB Elo.
        ranked_class = ' stats-table-ranked' if single else ''
        table = (f'<div class="table-scroll scroll-tall"><table class="stats-table t-{key}{ranked_class}" id="{table_id}" '
                 f'style="grid-template-columns:repeat({len(heads)},auto)"><thead><tr>')
        # Two labels per header, one visible at a time: tap sorts best-first,
        # tap again reverses (the visible label swaps to the other radio).
        table += ''.join(f'<th class="c{i}"><label for="s-{key}-{i}-a" class="sort-to-a" title="Sort best rank first">{escape(h)}</label>'
                         f'<label for="s-{key}-{i}-d" class="sort-to-d" title="Reverse the sort">{escape(h)}</label></th>'
                         for i, h in enumerate(heads, 1))
        body = ''.join(f'<tr style="{";".join(order)}">' + ''.join(cells) + '</tr>' for (cells, _), order in zip(rows, positions))
        return table + '</tr></thead><tbody>' + body + '</tbody></table></div>'

    def sort_css():
        """The radios every header label points at, plus the rules that turn
        a checked one into a row order. They sit ahead of everything else on
        the page because a ~ sibling selector only looks forward; display:none
        so tapping a label doesn't jump-scroll up to its radio."""
        radios, rules = [], []
        for key, count in sort_columns.items():
            for i in range(1, count + 1):
                radios += [f'<input type="radio" name="s-{key}" id="s-{key}-{i}-{d}" class="sort-radio">' for d in 'ad']
                up, down, table = f'#s-{key}-{i}-a:checked~*', f'#s-{key}-{i}-d:checked~*', f'.t-{key}'
                rules += [f'{up} {table} tbody tr{{--o:var(--a{i})}}', f'{down} {table} tbody tr{{--o:var(--d{i})}}',
                          f'{up} {table} .c{i} .sort-to-a{{display:none}}', f'{up} {table} .c{i} .sort-to-d{{display:inline}}',
                          f'{down} {table} .c{i} .sort-to-a:after{{content:" ↓"}}']
        return ''.join(radios) + '<style>' + ''.join(rules) + '</style>'

    def build_view(stats, id_suffix):
        stats = stats.drop_duplicates('team').sort_values('team')
        view = ''
        for section_label, unit in [('Offense', 'off'), ('Defense', 'def')]:
            # QB Elo isn't part of this grid anymore -- it's shown once,
            # above Offense, outside the Raw/Model toggle entirely (see
            # below): it doesn't change between the two, so there's no
            # second copy of it to toggle.
            tables = [(group_label, metric_table(stats, unit, metrics, f'{unit}-{group_label.lower()}-{id_suffix}'))
                     for group_label, metrics in STAT_GROUPS]
            # Stacked one per row (Passing, Rushing, Misc) -- each table
            # sizes to its own content (see .stats-table/.stats-grid CSS),
            # so a 3-column table doesn't end up as wide as a 6-column one.
            cells = ''.join(f'<div class="stats-cell"><h3>{label}</h3>{html}</div>' for label, html in tables if html)
            view += f'<h2>{section_label}</h2><div class="stats-grid">{cells}</div>'
        return view

    if season is not None and week is not None and lookback is not None:
        result = lookback_description(season, week, lookback)
    else:
        result = '<p class="muted">Pregame league snapshot · observed rates · (#1) is best. Click a stat to sort by rank.</p>'
    # QB Elo above everything, own row, never toggled between raw/model
    # (already recency-weighted regardless, so there's nothing to toggle).
    raw_stats = stats.drop_duplicates('team').sort_values('team')
    result += f'<h3>QB Elo</h3>{metric_table(raw_stats, "off", ["qb_elo"], "qb-elo", quarterback_column=True)}'
    if weighted_stats is not None and not weighted_stats.empty:
        # CSS-only toggle -- same :checked~sibling trick as the packet
        # bundle's tabs, no JS needed. Radios/labels/views must all be flat
        # siblings for the ~ combinator to reach across them.
        result += ('<input type="radio" name="stats-mode" id="stats-mode-raw" class="stats-mode-radio" checked>'
                  '<label for="stats-mode-raw" class="stats-mode-label">Raw</label>'
                  '<input type="radio" name="stats-mode" id="stats-mode-model" class="stats-mode-radio">'
                  '<label for="stats-mode-model" class="stats-mode-label">Model (recency-weighted)</label>'
                  f'<div id="stats-view-raw" class="stats-view">{build_view(stats, "raw")}</div>'
                  f'<div id="stats-view-model" class="stats-view">{build_view(weighted_stats, "model")}</div>')
    else:
        result += build_view(stats, 'raw')
    return sort_css() + result


# Longest side of every embedded logo. The source PNGs are 500x500, but no
# logo displays above 70px and the single-file packet embeds each one dozens
# of times (~670 embeds a week) -- at full size that made packet_*.html ~50 MB,
# twice Gmail's 25 MB attachment cap (pack_a_punch.py emails it); at 128px
# it's ~12 MB, still sharp on 2x screens everywhere but the 70px banners.
LOGO_PX = 128


@lru_cache(maxsize=None)
def logo_data(team):
    """Base64 PNG of a team's logo, downscaled to LOGO_PX; None if missing."""
    path = Path('data/logos') / f'{team}.png'
    if not path.exists():
        return None
    from PIL import Image
    image = Image.open(path).convert('RGBA')
    image.thumbnail((LOGO_PX, LOGO_PX), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, 'PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode()


def logo(team):
    encoded = logo_data(team)
    if encoded is None:
        return ''
    return f'<img class="logo" alt="{escape(team)} logo" src="data:image/png;base64,{encoded}">'


def pretty(feature):
    if feature.startswith('diff_'):
        return pretty(feature[len('diff_'):]) + ' · off − def'
    labels = {'fourth_down_%': '4th-down conversion', 'third_down_%': '3rd-down conversion',
              'pass_completion_%': 'Pass completion', 'series_success_%': 'Series success',
              'stuff_%': 'Runs stuffed', 'sack_%': 'Sacks / pass play',
              'qb_hit_%': 'QB hits / pass play', 'penalties_pp': 'Penalty flags / play',
              'first_down_pp': 'First downs / play', 'turnovers_pp': 'Turnovers / play',
              'explosive_run_%': 'Runs of 10+ yards', 'explosive_pass_%': 'Passes of 20+ yards',
              # Model 2.1's EPA family (nflverse expected points; success = a positive-EPA play).
              'pass_epa_pp': 'Pass EPA/play', 'run_epa_pp': 'Run EPA/play',
              'pass_success_%': 'Pass success rate', 'run_success_%': 'Run success rate',
              # two_sided_packet's per-dimension weather features -- context_weather
              # (below) is a different, single combined feature from an older model.
              'context_weather_feels_like_f': 'Feels like (°F)', 'context_weather_wind_mph': 'Wind (mph)',
              'context_weather_precip_inches': 'Precipitation (in)', 'context_weather_rain_inches': 'Rain (in)',
              'context_weather_snowfall_inches': 'Snowfall (in)', 'context_weather_snow_depth_inches': 'Snow depth (in)',
              'context_weather_indoor': 'Indoor'}
    if feature in labels:
        return labels[feature]
    label = feature.replace('away_off_', 'Away offense · ').replace('away_def_', 'Away defense · ')
    label = label.replace('total_', 'Combined · ').replace('_', ' ')
    for old, new in [('ypp', 'yards/play'), (' pp', '/play'), ('qb elo', 'QB Elo'),
                     ('home field adv', 'Home field'), ('away rest adv', 'Away rest advantage'),
                     ('away travel adv', 'Travel distance'),
                     ('away game importance', 'Game-importance difference')]:
        label = label.replace(old, new)
    return label[:1].upper() + label[1:]


def attribution(row):
    values = pd.Series({pretty(c[5:]): float(row[c]) for c in row.index if c.startswith('attr_')})
    order = values.abs().sort_values(ascending=False).index
    shown = values.loc[order[:8]].copy()
    if len(order) > 8:
        shown['Other features (net)'] = values.loc[order[8:]].sum()
    scale = max(shown.abs().max(), .01)
    bars = []
    for name, value in shown.items():
        width = 49 * abs(value) / scale
        left = 50 if value >= 0 else 50 - width
        bars.append(f'<div class="barrow"><span>{escape(name)}</span><div class="track"><span class="bar {"negative" if value < 0 else ""}" style="left:{left:.2f}%;width:{width:.2f}%"></span></div><span class="value">{value:+.2f}</span></div>')
    residual = float(row.prediction - row.baseline - values.sum())
    target = 'away − home margin' if row.get('market') == 'spread' else 'prediction'
    return ''.join(bars) + (f'<p class="muted">Baseline {row.baseline:+.2f} + features {values.sum():+.2f}'
                            f' + numerical residual {residual:+.4f} = {target} {row.prediction:+.2f} points.</p>')


def importance_bars(table):
    """table: feature (already pretty()'d)/importance/std, pre-sorted.
    Same .barrow/.track/.bar style as attribution() -- importance is
    usually positive (removing a feature hurt) but can go negative (a
    feature that was actively hurting predictions), so it's a diverging
    bar like attribution's, not a one-sided chart."""
    scale = max(table.importance.abs().max(), .01)
    bars = []
    for _, r in table.iterrows():
        width = 49 * abs(r.importance) / scale
        left = 50 if r.importance >= 0 else 50 - width
        bars.append(f'<div class="barrow"><span>{escape(r.feature)}</span><div class="track">'
                    f'<span class="bar {"negative" if r.importance < 0 else ""}" style="left:{left:.2f}%;width:{width:.2f}%"></span></div>'
                    f'<span class="value">{r.importance:+.3f}</span></div>')
    return ''.join(bars)


def team_table(row, stats):
    if stats.empty:
        return '<p class="muted">Team summaries unavailable.</p>'
    stats = stats.iloc[0]
    rows = []
    for unit, metric in [('off', 'run_ypp'), ('off', 'pass_ypp'), ('off', 'series_success_%'),
                         ('off', 'qb_elo'), ('def', 'pass_ypp'), ('def', 'run_ypp')]:
        values = [stats.get(f'{side}_raw_{unit}_{metric}', np.nan) for side in ['away', 'home']]
        formatted = [f'{v:.1%}' if '%' in metric and pd.notna(v) else f'{v:.2f}' if pd.notna(v) else '—' for v in values]
        name = ('Offense · ' if unit == 'off' else 'Defense · ') + pretty(metric)
        rows.append(f'<tr><td>{escape(name)}</td><td>{formatted[0]}</td><td>{formatted[1]}</td></tr>')
    return f'<table><tr><th>Pregame team rates</th><th>{escape(row.away_team)}</th><th>{escape(row.home_team)}</th></tr>{"".join(rows)}</table>'


@lru_cache(maxsize=8)
def team_records(season, week):
    """{team: '2-1'} as of kickoff -- every completed game earlier this season.
    Ties get a third number, the way a record is actually written."""
    sched = packet_schedule()
    played = sched[(sched.season == season) & (sched.week < week)
                   & sched.away_score.notna() & sched.home_score.notna()]
    tally = {}
    for game in played.itertuples():
        for team, own, other in [(game.away_team, game.away_score, game.home_score),
                                 (game.home_team, game.home_score, game.away_score)]:
            record = tally.setdefault(team, [0, 0, 0])
            record[0 if own > other else 1 if own < other else 2] += 1
    return {team: f'{w}-{l}' + (f'-{t}' if t else '') for team, (w, l, t) in tally.items()}


def game_header(row, market, action):
    def qb(side):
        name = row.get(f'{side}_qb_short')
        if pd.isna(name):
            name = row.get(f'{side}_qb_name')
        elo = row.get(f'{side}_raw_off_qb_elo')
        text = escape(str(name)) if pd.notna(name) else 'QB unavailable'
        return text + (f' · {elo:.1f} Elo' if pd.notna(elo) else '')

    away, home = escape(row.away_team), escape(row.home_team)
    sign = -1 if market == 'spread' else 1
    fmt = '+.1f' if market == 'spread' else '.1f'
    values = [format(sign * row.market_base, fmt), format(sign * row.prediction, fmt),
              f'{abs(row.edge):.1f}', f'{np.sqrt(row.variance):.1f}']
    headings = ['Market' if market == 'spread' else 'Market O/U',
                'Model' if market == 'spread' else 'Model O/U', 'Edge', 'SD']
    records = team_records(int(row.season), int(row.week)) if 'season' in row else {}

    def badge(team):
        record = records.get(team)
        return f'<span class="record">({escape(record)})</span>' if record else ''
    if action == 'PASS':
        pick = '–'
    else:
        tier = row.get('tier') or pick_tier(market, row.week, row.get('total_game_importance'),
                                           float(np.sqrt(row.variance)), row.get('market_base'),
                                           abs(float(row.edge)))
        pick = f'<span class="pill pick tier-{tier}" title="{escape(tier)} confidence">{escape(action)}</span>'
    scores = ''
    if pd.notna(row.get('away_points')) and pd.notna(row.get('home_points')):
        # Three cells (label | score | spacer) so the score itself sits dead
        # center under the numbers, with its label just to the left.
        label = 'Implied score:' if row.get('scores_implied', False) == True else 'Prediction:'
        scores = (f'<span class="score-label">{label}</span>'
                  f'<span class="score-value">{away} {row.away_points:.1f} - {row.home_points:.1f} {home}</span><span></span>')
    # Each number carries its own label directly above it, and the numbers sit
    # in one group between the teams instead of being stretched across the
    # full width by a grid row of their own.
    cells = ''.join(f'<div class="pick-cell"><span class="pick-label">{escape(h)}</span>'
                    f'<span class="pick-value">{v}</span></div>'
                    for h, v in zip(headings, values))
    cells += (f'<div class="pick-cell"><span class="pick-label">PICK</span>'
              f'<span class="pick-value">{pick}</span></div>')
    return ('<header class="pick-header"><div class="pick-grid">'
            + f'<div class="pick-team"><span class="pick-label">Away</span>'
              f'<span class="identity">{logo(row.away_team)}{away}{badge(row.away_team)}</span></div>'
            + f'<div class="pick-numbers">{cells}</div>'
            + f'<div class="pick-team home"><span class="pick-label">Home</span>'
              f'<span class="identity">{badge(row.home_team)}{home}{logo(row.home_team)}</span></div>'
            + f'<div class="pick-qb">{qb("away")}</div><div class="pick-score">{scores}</div>'
            + f'<div class="pick-qb home">{qb("home")}</div></div></header>')


def weather_details(row):
    """Right-hand text for the chart's Weather block: (kickoff for its
    heading, {weather input: the value it had}) -- e.g. 'Sun Sep 20 · 4:25 PM
    ET' and {'feels_like_f': '95°F', 'wind_mph': '3.6 mph', ...}. These are
    the model's own inputs, so an indoor game reads the fixed 72°F / calm /
    dry conditions the model uses there. The feels-like reading also names
    the roof when it isn't plain outdoors -- including when the schedule
    doesn't list one yet, which the model treats as outdoors."""
    def number(column):
        value = row.get(column)
        return float(value) if pd.notna(value) else None

    date = pd.to_datetime(row.get('gameday'), errors='coerce')
    time = pd.to_datetime(row.get('gametime'), format='%H:%M', errors='coerce')
    kickoff = ' · '.join(p for p in [f'{date:%a %b} {date.day}' if pd.notna(date) else None,
                                     f'{time:%I:%M %p} ET'.lstrip('0') if pd.notna(time) else None] if p)
    game = schedule_game(row)
    roof = row.get('roof') if pd.notna(row.get('roof')) else (game.get('roof') if game is not None else None)
    roof = str(roof).strip().lower() if roof is not None and pd.notna(roof) else ''
    indoor = roof in ('dome', 'closed')
    feels, air = number('feels_like_f'), number('temperature_f') if number('temperature_f') is not None else number('temp')
    reading = lambda value, fmt: format(value, fmt) if value is not None else '—'
    roof_note = {'dome': 'dome', 'closed': 'roof closed', 'open': 'roof open', 'outdoors': ''}.get(roof, 'roof not listed')
    values = {
        'feels_like_f': ' · '.join(p for p in [(f'{feels:.0f}°F' if feels is not None else '—')
                                               + (f' (air {air:.0f}°F)' if air is not None and not indoor else ''), roof_note] if p),
        'wind_mph': reading(number('wind_mph'), '.1f') + ' mph',
        'precip_inches': reading(number('precip_inches'), '.2f') + ' in',
    }
    return kickoff, {k: v.replace('— mph', '—').replace('— in', '—') for k, v in values.items()}


def specs_page(spec, running=None):
    """The Specs tab: what produced the numbers, and what the pick colours
    on the sheet actually mean.

    The tier guide leads, because it is the part a reader comes looking
    for; the rest is model_spec's spec for this run, the same content as
    the version folder's model.json.

    Nested dictionaries are flattened into one "parent · child" label
    rather than a definition list inside a definition list -- the spec is
    three deep in places, and indenting it that far turned a page of
    facts into a page of structure."""
    def label(key):
        return key.replace('_', ' ').capitalize()

    def leaf(value):
        if isinstance(value, list):
            short = all(len(str(v)) < 40 for v in value)
            return (escape(', '.join(map(str, value))) if short
                    else '<ul>' + ''.join(f'<li>{escape(str(v))}</li>' for v in value) + '</ul>')
        return escape(str(value))

    def rows(value, prefix=''):
        """(label, html) pairs, one per fact, however deep it was nested."""
        out = []
        for key, item in value.items():
            name = f'{prefix} · {label(key)}' if prefix else label(key)
            if isinstance(item, dict):
                out.extend(rows(item, name))
            else:
                out.append((name, leaf(item)))
        return out

    def section(key, value):
        body = ('<dl class="spec-list">'
                + ''.join(f'<dt>{escape(name)}</dt><dd>{html}</dd>' for name, html in rows(value))
                + '</dl>') if isinstance(value, dict) else f'<p class="spec-prose">{leaf(value)}</p>'
        return f'<section class="card spec-card"><h2>{escape(label(key))}</h2>{body}</section>'

    model = spec['model']
    head = (f'<h1>{escape(model["name"])} {escape(model["version"])}</h1>'
            f'<p class="report-date">Code {escape(model["code"])} · generated {escape(model["generated"])} · '
            'also saved as ../model.json</p>')
    return (head + tier_guide(running, standalone=True)
            + ''.join(section(key, value) for key, value in spec.items() if key != 'model'))


def packet_tabs(active):
    return '<nav class="packet-tabs" aria-label="Weekly report">' + ''.join(
        f'<a href="{path}"{""" aria-current="page" """ if key == active else ""}>{label}</a>'
        for key, path, label in [('headline', 'index.html', 'Headline'), ('spread', 'spread.html', 'Spread'),
                                ('total', 'total.html', 'Totals'), ('importance', 'importance.html', 'Feature importance'),
                                ('stats', 'stats.html', 'Stats'), ('specs', 'specs.html', 'Specs')]) + '</nav>'


def bundle_single_file(folder):
    """Bundle embedded pages, assets and downloads into one portable report."""
    folder = Path(folder)
    pages = [('headline', 'Headline', 'index.html'), ('spread', 'Spread', 'spread.html'),
             ('total', 'Totals', 'total.html'), ('importance', 'Feature importance', 'importance.html'),
             ('stats', 'Stats', 'stats.html'), ('specs', 'Specs', 'specs.html')]
    tabs, panels, title = [], [], 'Weekly packet'
    for key, label, name in pages:
        path = folder / name
        if not path.exists():
            continue
        html = path.read_text(encoding='utf-8')
        if key == 'headline':
            found = re.search(r'<title>([^<]*)</title>', html)
            title = found.group(1) if found else title
        main = re.search(r'<main class="([^"]*)">(.*)</main>', html, re.DOTALL)
        theme, content = (main.group(1), main.group(2)) if main else ('', html)
        content = re.sub(r'<nav class="packet-tabs".*?</nav>', '', content, flags=re.DOTALL)
        # Embed CSV downloads; remove cross-file links instead of shipping broken ones.
        def local_link(match):
            href, text = match.groups()
            asset = (folder / href).resolve()
            if asset.parent == folder.resolve() and asset.suffix == '.csv' and asset.is_file():
                encoded = base64.b64encode(asset.read_bytes()).decode()
                return f'<a download="{escape(asset.name)}" href="data:text/csv;base64,{encoded}">{text}</a>'
            return text
        content = re.sub(r'<a\b[^>]*href="([^"]*)"[^>]*>(.*?)</a>', local_link, content, flags=re.DOTALL)
        content = re.sub(r'<iframe\b.*?</iframe>', '', content, flags=re.DOTALL)
        checked = ' checked' if not tabs else ''
        tabs.append(f'<input type="radio" name="pkgtab" id="pt-{key}" class="pkgtab-radio"{checked}>'
                    f'<label class="pkgtab-label" for="pt-{key}">{escape(label)}</label>')
        panels.append(f'<main class="pkgpanel {escape(theme)}" data-tab="{key}">{content}</main>')
    if not panels:
        return None
    out = folder / 'packet.html'
    out.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8">'
                   f'<meta name="viewport" content="width=device-width"><title>{escape(title)}</title>'
                   f'{font_license()}<style>{font_style()}{STYLE}</style><body><div class="packet-bundle">'
                   f'{"".join(tabs)}{"".join(panels)}</div></body></html>', encoding='utf-8')
    return out


# Checked on Model 2.0's own backtest (data/bt/model_2.0/2020-2025, 1693
# games, six seasons) rather than the single 2024/2025 calibration these came
# from under the previous model.
#   total: edge >= 5, no SD condition -- 54.7% on 522 bets (+5.2% roi), at or
#          above 51% in every one of the six seasons. The old SD <= 4.43
#          condition made it worse, not better: totals with SD <= 4.0 won
#          49.1% while SD >= 4.5 won 53.2%, so requiring low SD cut the rule
#          to 53.2%. Its 95% roi interval still spans zero ([-2.6%, +13.0%])
#          and the last two seasons are 53%/51%, i.e. break-even -- a mild
#          preference, not a proven edge.
#   spread: no rule held up. The old 5.0/4.83 cutoff went 50.6% (-2.7%) across
#          the six seasons, and a search over 1230 rule variants produced
#          nothing that survived out of sample -- the same search on shuffled
#          outcomes averages +9% roi, so anything below that is luck. The
#          model's disagreement with the spread market carries almost no
#          information (b = +0.07, t = 1.1).
HIGH_CONFIDENCE_CUTOFFS = {
    'spread': dict(diff_cutoff=2.0, sd_cutoff=None),
    'total': dict(diff_cutoff=5.0, sd_cutoff=None),
}


# Confidence tiers. A bucket's letter is measured, not assigned: each bucket
# below carries the hit rate it posted on ONE backtest -- Model 2.0 with
# 'weighted' stats and a 20-week lookback, 2010-2025, 4363 games (TIER_SOURCE)
# -- and TIER_BANDS turns that rate into S, A or B. Re-measure after a new
# backtest, edit the rate, and the letter follows by itself.
#
# A bucket under B's floor is not a weak pick, it is no pick: the sheet prints
# a dash and the game is not graded (see NO_PICK). Break-even at -110 is 52.4%,
# so B's floor sits a point and a half above water, not at it.
#
# The letter is per-pick quality -- what to trust when deciding one game. What
# a bucket is WORTH is a different number, and it is the one that pays: hit
# rate times how many picks the bucket throws a season. 'worth' carries it, at
# one unit a bet and -110 where the real price is missing. Every bucket here
# runs 7-14 picks a season, so per-pick quality still drives the ranking; a
# thinner bucket at a higher rate would not, which is why both are shown.
#
# What separates the buckets is the situation, not the model's own confidence:
# how late in the season it is (in September the 20-week lookback is mostly
# last season), whether the ensemble agrees with itself (spread), and where the
# market priced the total (totals).
#
# Every rate here comes from TIER_SOURCE and nothing else -- no averaging
# across runs. Each bucket's 'siblings' field is the same bucket measured on
# the other three completed backtests (carryover, steep-lookback, and Model
# 2.1 with importance weighting). Those are different models, not re-runs, so
# they do not set the letter; they are there because a bucket its siblings
# disagree with is a bucket to size down. Where they spread wide, the shipped
# rate is probably the optimistic end.
TIER_MODEL = 'Model 2.0 · weighted'   # the model the packet builds today
# A different architecture, kept defined because the packet can still run it
# (model_spec.PROFILES) -- but it has no buckets. Its weeks 1-12 spread band
# read 57.9% on its first three seasons, 54.9% at 85% of the backtest and
# 53.3% on the finished one: below the floor, CI [50.0, 56.7]. The mechanism
# was never a matchup read. Its predictions have a standard deviation of 3.1
# against an actual margin spread of 14.5, correlate 0.16 with results where
# the two-sided model manages 0.31, and favour the home team 67% of the time,
# so a market line away from its near-constant output became an "edge" and
# the band picked the underdog 90% of the time. Backing the dog blind on the
# same games returns 51.9%, so the model added about a point and a half --
# real, but not enough to bet.
SHARED_MODEL = 'Model 2.1 · shared'
SOLVED_MODEL = 'Model 2.0 · solved'
# A bucket is measured on ONE model and only fires on that model. This is not
# bookkeeping: the weeks 1-12 spread band below is 54.9% on the shared model
# and 47-49% on all five two-sided runs, so applying it to the wrong one turns
# an edge into a leak. pick_bucket() filters on this field.
# Playoff leverage (playoff_importance.game_importance, 0-1): the top quartile
# of week 13+ games, which is where the spread edge switches on.
LEVERAGE = 0.70
TIER_SOURCE = f'{TIER_MODEL}, 20-week lookback, 2010-2025 backtest: 4363 games'
# Measured hit-rate floors. C is the new bottom rung: it clears break-even
# at -110 (52.38%) but not by much, so it is a bet rather than a lean and
# the sheet says so by colour. Anything under C is not a pick at all.
TIER_BANDS = [('S', .610), ('A', .580), ('B', .550), ('C', .524)]
DEAD_TOTAL = (42.0, 46.0)
PICK_BUCKETS = {
    'spread': dict(edge=2.0, buckets=[
        dict(rule=f'week 13 on, including the playoffs, in a game that still matters '
                  f'(playoff leverage at least {LEVERAGE:.2f})',
             test=lambda week, importance, sd, line, edge: (week >= 13 and importance is not None
                                                            and pd.notna(importance) and importance >= LEVERAGE),
             model=TIER_MODEL, rate=.607, n=252, worth='+2.5u a season on ~16 picks, up 12 of 16 years',
             frequency=f'16 a season · about 1.6 a week across the 10 weeks it can fire',
             eras='59 / 58 / 59 / 65%',
             siblings='carryover 60.1%, steep 59.1%, 2.1-importance 60.2%',
             note='Playoff leverage, not the ensemble\'s own confidence, is what separates a December spread '
                  'the model can read from one it cannot. From week 13 on, the top leverage quarter hits '
                  '60.7% while the other three sit at 49-52% -- a step, not a slope. The same split in weeks '
                  '1-12 is worthless, and the high-leverage quarter is actually the worst of them there '
                  '(46.6%), which fits: in September nothing is decided, so the number measures nothing. '
                  'This replaced an SD-based rule that scored the same here but fell to 52.4% and 49.4% on '
                  'two of the siblings; this one holds at 59-60% on all four, worst era 58%.'),
    ]),
    'spread_solved': dict(edge=2.0, buckets=[
        dict(rule='weeks 1-5, the model off the spread by 4 or more, ensemble SD at most 4.0',
             test=lambda week, importance, sd, line, edge: (week <= 5 and edge is not None
                                                            and abs(edge) >= 4
                                                            and sd is not None and sd <= 4.0),
             model=SOLVED_MODEL, rate=.585, n=200, worth='+1.8u a season on ~15 picks, up 10 of 13 years',
             frequency=f'15 a season · about 3.1 a week across the 5 weeks it can fire',
             eras='55 / 60 / 65 / 56%',
             siblings='not yet measured on other runs',
             note='The first spread edge before week 13, and it came from fixing an input rather than '
                  'finding a filter. The decay presets price a prior-season game at their 0.05 floor by '
                  'September; taper_solver.py fits 0.40, so in week 2 the default model reads a 5.0 '
                  'yards-per-play team that had one 15.0 game as a 10.4 team. Correcting that lifts weeks '
                  '1-5 spread IC from 0.239 to 0.292 on the same 1011 games. Two honest caveats: paired '
                  'against the default on identical games the pick-level gain is 62-50 on 112 '
                  'disagreements (McNemar p = 0.30), and the shuffled-outcome null beats 58.5% in 18 of '
                  '400 tries (p = 0.045) -- it passed, not comfortably. Weeks 6-12 are still backtesting; '
                  'if the edge extends, this window widens.'),
    ]),
    'total': dict(edge=5.0, buckets=[
        # Ordered, and each rate is measured on what the buckets above it
        # leave behind -- a rule that looks strong standalone can be living
        # off games an earlier bucket already takes. Weeks 1-4 with a 7-point
        # differential reads 56.9% on its own and 53.2% on the residual, which
        # is why September has no bucket of its own despite looking like it
        # deserved one. Found with alpha_juicer.py (10,500 combinations of
        # differential, SD, leverage and week bands, scored on all five
        # completed backtests at once).
        dict(rule='week 9 on, posted total outside 42-46, ensemble SD at most 4.0',
             test=lambda week, importance, sd, line, edge: (week >= 9 and not dead_total(line)
                                                            and (sd is None or sd <= 4.0)),
             model=TIER_MODEL, rate=.683, n=145, worth='+2.8u a season on ~9 picks, up 12 of 16 years',
             frequency=f'9 a season · about 0.6 a week across the 14 weeks it can fire',
             eras='67 / 65 / 71 / 68%',
             siblings='carryover 61.8%, steep 62.8%, 2.1-importance 67.7% (n=31), 2.2-travel 66.4%',
             note='The best bucket on the board and the flattest -- no era below 65%, halves 66/69%. What it '
                  'adds over the same weeks without the SD cut (59.8%) is the ensemble agreeing with itself, '
                  'the one condition that survived on the totals side after it failed on the spread side.'),
        dict(rule='weeks 1-12, the model off the total by 8 or more, ensemble SD at most 4.5',
             test=lambda week, importance, sd, line, edge: (week <= 12 and edge is not None
                                                            and abs(edge) >= 8 and (sd is None or sd <= 4.5)),
             model=TIER_MODEL, rate=.634, n=101, worth='+1.3u a season on ~6 picks, up 11 of 16 years',
             frequency=f'6 a season · about 0.5 a week across the 12 weeks it can fire',
             eras='– / 70 / 71 / 54%',
             siblings='carryover 58.0%, steep 55.4%, 2.1-importance 52.5% (n=40), 2.2-travel 59.0%',
             note='The one thing that works before week 9: not a better read on ordinary games, but the rare '
                  'ones where the model is a touchdown-plus away from the market and its own runs agree. It '
                  'covers September too, where nothing else does. The size of the disagreement is doing the '
                  'work -- the same weeks at 5-7 points are 52-55%.'),
        dict(rule='weeks 5-8, posted total outside 42-46',
             test=lambda week, importance, sd, line, edge: 5 <= week <= 8 and not dead_total(line),
             model=TIER_MODEL, rate=.603, n=136, worth='+1.3u a season on ~9 picks, up 9 of 16 years',
             frequency=f'8 a season · about 2.1 a week across the 4 weeks it can fire',
             eras='55 / 60 / 61 / 65%',
             siblings='carryover 60.0%, steep 51.5%, 2.1-importance 53.8%, 2.2-travel 60.6%',
             note='The model’s edge in the stretch the spread side cannot touch at all. The carryover and '
                  'travel runs match it; the steep and 2.1 runs do not, so the stat-weighting choice matters '
                  'more here than anywhere else on the board.'),
        dict(rule='week 9 on, posted total outside 42-46, ensemble SD above 4.0',
             test=lambda week, importance, sd, line, edge: week >= 9 and not dead_total(line),
             model=TIER_MODEL, rate=.543, n=258, worth='+0.6u a season on ~16 picks, up 10 of 16 years',
             frequency=f'16 a season · about 1.2 a week across the 14 weeks it can fire',
             eras='58 / 53 / 49 / 57%',
             siblings='carryover 56.0%, steep 55.4%, 2.1-importance 52.3%, 2.2-travel 54.7%',
             note='The other half of the SD split, kept rather than dropped because it holds on every run, '
                  'but it is the thinnest thing on the board: one era (2018-21) sits at 49%, so this is a '
                  'bucket to bet flat and not to press.'),
    ]),
}
# What the buckets deliberately leave out, and what it would have cost.
NO_PICK = {
    'spread_solved': dict(model=SOLVED_MODEL,
                   rule='every other weeks 1-5 spread: a smaller disagreement, or the ensemble not '
                        'agreeing with itself',
                   record='51.6% of 459 would-be picks, -1.4% per bet', worth='-0.5u a season on ~35 picks',
                   eras='50 / 54 / 44 / 58%',
                   note='Fixing the taper lifts the whole early-season slate, but only the corner where the '
                        'model both disagrees loudly and agrees with itself clears the bar. The rest is '
                        'a point above break-even before vig and a point below it after -- which is still '
                        'a different animal from the default model here, where the same games are 50.1%.'),
    'spread': dict(model=TIER_MODEL, rule='every other spread: any week before 13, and late games with '
                                          'nothing riding on them',
                   record='49.8% of 2599 would-be picks, -4.9% per bet', worth='-8.0u a season on ~162 picks',
                   eras='49 / 48 / 51 / 51%',
                   note='No era above break-even, and not for want of looking. A gated sweep of every '
                        'combination of week window, edge, ensemble SD, playoff leverage and line size -- '
                        'about 1,800 rules, re-scored on all four backtests -- clears 54 rules here against '
                        'the 59 that the same gate clears on coin-flip outcomes. The spread search finds no '
                        'more than chance does, so everything outside the bucket above stays unbet.'),
    'total': dict(model=TIER_MODEL, rule='everything else: weeks 1-4 without a big differential, the 42-46 '
                                         'band, and any week the buckets above do not claim',
                  record='46.3% of 395 would-be picks, -11.5% per bet', worth='-2.9u a season on ~25 picks',
                  eras='45 / 49 / 50 / 42%',
                  note='No era anywhere near break-even. Two dead spots pooled: September, when the 20-week '
                       'lookback is mostly last season and only a huge disagreement means anything, and the '
                       '42-46 band, where this model is wrong in every era.')
}
# S regal purple, A green, B light yellow, C light orange. B and C are
# the close pair -- a light yellow and a light orange sit about 20 degrees
# apart, so C is pushed as far toward orange as 'light' allows.
TIER_COLORS = {'S': '#d9b3ff', 'A': '#b9e4c4', 'B': '#fff2b8', 'C': '#ffcb99'}
BAND_LABELS = {'S': '61%+', 'A': '58-61%', 'B': '55-58%', 'C': '52.4-55%'}


def dead_total(line):
    """The 42-46 band on the posted total, where this model has no edge."""
    return line is not None and pd.notna(line) and DEAD_TOTAL[0] <= float(line) <= DEAD_TOTAL[1]


def tier_band(rate):
    """The letter a measured hit rate earns, or None below the bottom band."""
    for name, floor in TIER_BANDS:
        if rate >= floor:
            return name
    return None


def market_key(market, running=None):
    """Which bucket list serves this market for the model in hand. A profile
    with buckets of its own gets its own list ('{market}_shared'); when it has
    none -- the case today -- this is the identity, and pick_bucket's model
    filter then finds nothing, which is the right answer for a model with no
    measured edge."""
    own = {SHARED_MODEL: f'{market}_shared',
           SOLVED_MODEL: f'{market}_solved'}.get(running or TIER_MODEL)
    return own if own in PICK_BUCKETS else market


def pick_bucket(market, week, importance=None, sd=None, line=None, edge=None, running=None):
    """The first bucket this pick falls in, or None if none covers it. A
    bucket measured on another model never applies -- see SHARED_MODEL."""
    model = running or TIER_MODEL
    for bucket in PICK_BUCKETS[market_key(market, running)]['buckets']:
        if bucket['model'] == model and bucket['test'](int(week), importance, sd, line, edge):
            return bucket
    return None


def pick_tier(market, week, importance=None, sd=None, line=None, edge=None, running=None):
    """This pick's tier (S, A or B), or None when no bucket covers it -- which
    means it is not a pick at all; see NO_PICK. sd: the ensemble's
    disagreement. line: the market's own number (the posted total, or the
    spread). Either being None never blocks a bucket."""
    bucket = pick_bucket(market, week, importance, sd, line, edge, running)
    return tier_band(bucket['rate']) if bucket else None


def apply_tiers(frame, market, running=None):
    """Tier every settled row, and let the tier decide what counts as a pick:
    an edge big enough to clear the threshold is necessary but not sufficient,
    because outside the buckets this model loses money (NO_PICK). Rows that no
    bucket covers keep qualifies=False and print a dash."""
    data = frame.copy()
    variance = data.variance if 'variance' in data else pd.Series(np.nan, index=data.index)
    sd = np.sqrt(variance.clip(lower=0))
    importance = (data.total_game_importance if 'total_game_importance' in data
                  else pd.Series(np.nan, index=data.index))
    data['tier'] = [pick_tier(market, week, imp, None if pd.isna(s) else float(s), line, abs(edge), running)
                    for week, imp, s, line, edge in zip(data.week, importance, sd, data.market_base, data.edge)]
    data['qualifies'] = data.qualifies & data.tier.notna()
    return data


def tier_guide(running=None, standalone=False):
    # running: the model behind one market, or {market: model} when the packet
    # carries two -- each section is judged against ITS market's model, not
    # the packet's first one.
    """The expandable key under the sheet: every bucket, the run it was
    measured on, its rate, and the band that rate earns it.

    running: the model this packet was actually built with. If it isn't the
    one the buckets were measured on, the guide says so at the top -- the
    rates are model-specific (the spread S bucket alone swings from 60.0% to
    49.4% across runs), so a mismatch makes them indicative at best."""
    behind = running if isinstance(running, dict) else {m: running for m in ['spread', 'total']}
    sections = []
    for market, spec in PICK_BUCKETS.items():
        owner = spec['buckets'][0]['model']
        # 'spread_solved' / 'total' -> 'spread' / 'total': the suffix names
        # which model owns the bucket set, not a different market.
        serves = market.split('_')[0]
        live = behind.get(serves) in (None, owner)
        rows = ''
        for bucket in sorted(spec['buckets'], key=lambda b: -b['rate']):
            tier = tier_band(bucket['rate'])
            rows += (f'<tr><td><span class="tier-chip" style="background:{TIER_COLORS[tier]}">{tier}</span></td>'
                     f'<td>{escape(bucket["rule"])}</td>'
                     f'<td class="tier-record">{escape(bucket["model"])}</td>'
                     f'<td class="tier-record">{100 * bucket["rate"]:.1f}% of {bucket["n"]} picks</td>'
                     f'<td class="tier-record">{escape(bucket["worth"])}</td>'
                     f'<td class="tier-record">{escape(bucket.get("frequency", "—"))}</td>'
                     f'<td class="tier-record">{escape(bucket["eras"])}</td>'
                     f'<td class="tier-record">{escape(bucket["siblings"])}</td></tr>'
                     f'<tr class="tier-note-row"><td></td><td colspan="7">{escape(bucket["note"])}</td></tr>')
        skip = NO_PICK[market]
        rows += (f'<tr><td><span class="tier-chip tier-chip-none">–</span></td>'
                 f'<td>{escape(skip["rule"])}</td><td class="tier-record">{escape(skip["model"])}</td>'
                 f'<td class="tier-record">{escape(skip["record"])}</td>'
                 f'<td class="tier-record">{escape(skip["worth"])}</td><td class="tier-record">—</td>'
                 f'<td class="tier-record">{escape(skip["eras"])}</td><td class="tier-record">—</td></tr>'
                 f'<tr class="tier-note-row"><td></td><td colspan="7">{escape(skip["note"])}</td></tr>')
        sections.append(
            f'<h4>{escape(serves.title())} picks · {escape(owner)}'
            + ('' if live else ' <em>(not the model behind this column — shown for reference, '
               'never fires here)</em>') + '</h4>'
            f'<p class="tier-qualify">A pick needs the model to disagree with the market by at least '
            f'{spec["edge"]:g} points, and to land in one of these buckets.</p>'
            '<table class="tier-table"><thead><tr><th></th><th>Bucket</th><th>Model</th><th>Measured</th>'
            '<th>Worth</th><th>How often</th><th>By era</th><th>Other runs</th></tr></thead><tbody>'
            + rows + '</tbody></table>')
    gaps = ('<p class="tier-gap">Spreads have no bucket in weeks 6-12, and that is a finding rather '
            'than an oversight. Five independent searches have looked: an exhaustive hit-rate grid, an '
            'error-surface search over the same dimensions, the same search with absolute rather than '
            'percentile cuts, a nine-arm sweep of the sided-spread architecture, and a gate-first search '
            'that added the size of the posted line. Three cells in all of that cleared the era and '
            'halves gates, and all three fell apart on a second model (worst sibling 47-51%) while '
            'paying about half a unit a season. The pattern fits: weeks 1-3 are exploitable because the '
            'model is carrying stale priors, week 13 on because playoff leverage distorts what teams are '
            'playing for, and weeks 6-12 is the stretch where the model has plenty of information and so '
            'does the market. The one thing that has ever opened a closed window here was changing the '
            'model, not filtering it harder. Totals are covered in every week.</p>')
    bands = ', '.join(f'{tier} = {BAND_LABELS[tier]}' for tier, _ in TIER_BANDS)
    bottom = ('<p class="tier-note">All six buckets together: about 59 picks a season for +7.6 units at one '
              'unit a bet, up in 15 of the 16 backtested seasons (worst -4.0u in 2014, best +18.2u in 2023). '
              'The two S buckets in weeks 13-14 and the playoffs are half of that on a quarter of the picks, '
              'and the thinnest bucket is 8% of it. A bucket only pays what its hit rate earns times how '
              'often it fires, so a high rate on few games is worth less than it looks.</p>')
    # Only worth warning about when a column's model has no buckets of its
    # own: then the rates on show were measured on something else entirely.
    owners = {spec['buckets'][0]['model'] for spec in PICK_BUCKETS.values()}
    orphans = sorted({model for model in behind.values() if model and model not in owners})
    mismatch = ''
    if orphans:
        mismatch = (f'<p class="tier-warn">This packet was built with {escape(", ".join(orphans))}, which has no '
                    'buckets of its own — every rate below was measured on a different model. The buckets still '
                    'decide the picks, but their hit rates do not describe this one; see "Other runs" for how '
                    'far they move.</p>')
    open_tag, close_tag = (('<section class="card spec-card tier-guide"><h2>Pick tiers</h2>', '</section>')
                           if standalone else
                           ('<details class="tier-guide"><summary>What the pick colours mean</summary>',
                            '</details>'))
    return (open_tag + mismatch + gaps
            + f'<p class="tier-qualify">The letter is the bucket’s measured hit rate, not a judgement: '
            f'{bands}, and under {100 * TIER_BANDS[-1][1]:g}% is no pick at all.</p>'
            + ''.join(sections) + bottom
            + f'<p class="tier-note">Every rate is measured on the {escape(TIER_SOURCE)} — the run this packet '
              'builds — and nothing is averaged across models. "Other runs" is the same bucket on the carryover, '
              'steep-lookback and Model 2.1 backtests: different models, shown so you can see which buckets '
              'survive a change of model and which do not. Break-even at -110 is 52.4%. The dash row is not a weak pick, it is no '
              'pick: those games lose money, so the sheet leaves them blank rather than grading them. Hit rates '
              'are what a bucket did historically, not a forecast for any single pick.</p>' + close_tag)


def running_model(folder, market=None):
    """Which model wrote this packet's `market` pages -- 'Model 2.0 · weighted'
    or 'Model 2.1 · shared'. Read per market, because a packet can carry two:
    weeks 1-12 spreads come from the shared architecture and everything else
    from the two-sided one (shared_scoring.SHARED_SPREAD_WEEKS). With no
    market, the first config found answers, which is what a whole-packet
    question (the tier guide's heading) wants. None if the folder predates
    these configs or came from another pipeline."""
    for name in ([market] if market else ['spread', 'total']):
        path = Path(folder) / f'{name}_config.json'
        if not path.exists():
            continue
        try:
            config = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if config.get('tier_model'):        # written explicitly since the packet runs two models
            return str(config['tier_model'])
        label = str(config.get('model', '')).split(' · ')[0].strip()
        preset = config.get('feature_calculation') or config.get('calculation')
        if label and preset:
            return f'{label} · {preset}'
    return None


def tier_legend(markets=('spread', 'total')):
    """The small colour key in the corner of the sheet: the bands themselves,
    since a letter now means one thing in both markets."""
    items = ''.join(f'<span class="tier-key"><i style="background:{TIER_COLORS[tier]}"></i>'
                    f'{escape(tier)} {escape(BAND_LABELS[tier])}</span>' for tier, _ in TIER_BANDS)
    # Provenance and the break-even line live on the Specs tab. Under the
    # sheet they were a paragraph of footnote between the reader and the
    # picks; the swatches alone say what the colours mean.
    return (f'<div class="tier-legend"><strong>Hit rate</strong>{items}'
            f'<span class="tier-key"><i class="tier-swatch-none"></i>– under '
            f'{100 * TIER_BANDS[-1][1]:g}%, no bet</span></div>')


def copy_picks_widget(season, week, light_table):
    """A "Copy" button for a picture of the picks table (picks_png, drawn
    in Python when the packet is built, so it exists no matter where the
    file is opened). Clicking it puts that PNG on the clipboard where the
    browser allows it -- paste straight into a chat, email or doc -- and a
    short confirmation fades in next to the button (CSS animation, restarted
    via a reflow trick so it still fires on a second click).

    Everywhere else -- phone previews that run no scripts, browsers that
    refuse clipboard images -- the button is really a label for a hidden
    checkbox: it reveals the picture itself, which any browser lets you
    right-click (desktop) or long-press (phone) to copy or save. When a
    script copy succeeds, the click is cancelled before the checkbox flips."""
    # Season and week only. Which model produced which column is recorded
    # in models.json and named per bucket in the tier guide; on the sheet
    # itself it was a line of machinery above the thing you came to read.
    subtitle = f'{int(season)} Week {int(week)}'
    legend = [(TIER_COLORS[tier], f'{tier}  hit rate {BAND_LABELS[tier]}') for tier, _ in TIER_BANDS]
    legend.append(('#ffffff', f'–  under {100 * TIER_BANDS[-1][1]:g}%, no bet'))
    png, width = picks_png(light_table, 'Model', subtitle, legend=legend)
    return ('<input type="checkbox" id="copy-picks-toggle" class="copy-toggle">'
           '<div class="copy-picks"><label for="copy-picks-toggle" class="copy-btn" onclick="copyPicksImage(event)">'
           '<span class="copy-open">Copy</span><span class="copy-close">Done</span></label>'
           '<span class="copy-hint">Right-click (or long-press) the picture to copy it</span>'
           '<span id="copy-feedback" class="copy-feedback">Copied as an image</span></div>'
           f'<img id="picks-png" class="copy-png" width="{width}" alt="Model picks, {escape(subtitle)}" '
           f'src="data:image/png;base64,{base64.b64encode(png).decode()}">'
           f'<script>{PICKS_COPY_SCRIPT}</script>')


def picks_png(light_table, title, subtitle, legend=(), scale=2):
    """(PNG bytes, CSS width) of the picks table, drawn with Pillow from the
    same light-theme Styler HTML headline_table(light=True) produces -- same
    cells, gradient/tier colors (read from the Styler's own <style> rules)
    and logos -- following .headline-table-light's CSS: Graduate 12px (14px
    team columns), 2px/6px padding, 1px #ddd grid, #f2f2f2 headers,
    right-aligned figures -- then the confidence key underneath. Fonts ship with the repo (Graduate) and
    with matplotlib (DejaVu Sans), so it renders the same on every OS.
    Drawn at 2x and tagged 144 dpi so it stays sharp when pasted."""
    from html.parser import HTMLParser
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    class Cells(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows, self.cell, self.css, self.in_style = [], None, '', False

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'style':
                self.in_style = True
            elif tag == 'tr':
                self.rows.append([])
            elif tag in ('th', 'td'):
                self.cell = dict(head=tag == 'th', id=attrs.get('id', ''), text='', logo=None)
                self.rows[-1].append(self.cell)
            elif tag == 'img' and self.cell is not None:
                self.cell['logo'] = (attrs.get('alt') or '').removesuffix(' logo')

        def handle_endtag(self, tag):
            if tag == 'style':
                self.in_style = False
            elif tag in ('th', 'td'):
                self.cell = None

        def handle_data(self, data):
            if self.in_style:
                self.css += data
            elif self.cell is not None:
                self.cell['text'] += data

    parsed = Cells()
    parsed.feed(light_table)
    # The Styler writes every data-dependent style (gradients, pick
    # highlight) into its <style> block as "#cell_id, #cell_id {prop: value}".
    styles = {}
    for selectors, body in re.findall(r'([^{}]+)\{([^}]*)\}', parsed.css):
        declarations = {k.strip(): v.strip() for k, v in (d.split(':', 1) for d in body.split(';') if ':' in d)}
        for selector in selectors.split(','):
            styles.setdefault(selector.strip().lstrip('#'), {}).update(declarations)

    faces = dict(sans=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans')),
                 bold=font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold')),
                 graduate=str(Path(__file__).parent / 'assets/fonts/Graduate-Regular.ttf'))
    fonts = {}

    def font(face, size):
        if (face, size) not in fonts:
            fonts[face, size] = ImageFont.truetype(faces[face], round(size * scale))
        return fonts[face, size]

    # Derived from HEADLINE_COLUMNS, not written out again: these were
    # hardcoded here and in the CSS separately, and had already drifted
    # (the home-logo column was being right-aligned and the last figure
    # column was not).
    small_columns = set(headline_indices(HEADLINE_SMALL, base=0))
    right_columns = set(headline_indices(HEADLINE_RIGHT, base=0))
    rows = []
    for row in (r for r in parsed.rows if r):
        styled = []
        for column, cell in enumerate(row):
            rule = styles.get(cell['id'], {})
            bold = cell['head'] or rule.get('font-weight') in ('600', '700', 'bold')
            quiet = not cell['head'] and column in small_columns
            size = 10.5 if quiet else 12
            styled.append(dict(
                cell, text=cell['text'].strip(), size=size,
                # The whole sheet is Graduate, which has one weight -- browsers
                # fake bold with a thicker stroke, so do the same here.
                font=font('graduate', size), stroke=bold,
                fill=rule.get('background-color', '#f2f2f2' if cell['head'] else '#ffffff'),
                color=rule.get('color', '#222222'),
                right=not cell['head'] and column in right_columns))
        rows.append(styled)

    # Padding is charged per column, and nineteen columns of it is most of
    # the sheet's width. Columns that hold one small thing -- a crest, a
    # rating that belongs to the name beside it -- get only what they need.
    pad_y, grid, logo_px = 2 * scale, scale, 16 * scale
    narrow = set(headline_indices(HEADLINE_TIGHT | {'away_elo', 'home_elo'}, base=0))
    # A rating sits tight against its quarterback; the gap belongs on the
    # far side so the two read as one field.
    snug = set(headline_indices({'away_qb', 'home_qb'}, base=0))

    def padding(column):
        if column in narrow:
            return 2 * scale
        if column in snug:
            return 3 * scale
        return 6 * scale
    columns = max(len(row) for row in rows)
    widths = [0] * columns
    for row in rows:
        for column, cell in enumerate(row):
            content = logo_px if cell['logo'] else cell['font'].getlength(cell['text'])
            widths[column] = max(widths[column], int(content + 0.999) + 2 * padding(column))
    heights = [max(max(round(cell['size'] * 1.25 * scale), logo_px if cell['logo'] else 0) for cell in row) + 2 * pad_y
               for row in rows]

    margin = 16 * scale
    legend_font = font('sans', 9)
    title_font, subtitle_font = font('bold', 18), font('graduate', 13)
    title_h, subtitle_h = round(18 * 1.2 * scale), round(13 * 1.25 * scale)
    table_top = margin + title_h + 2 * scale + subtitle_h + 10 * scale
    table_w = sum(widths) + (columns + 1) * grid
    table_h = sum(heights) + (len(rows) + 1) * grid
    # The key reads across in one line rather than stacking: four short
    # entries down the right-hand side left a tall empty block beside them,
    # and a row of swatches is quicker to scan anyway.
    swatch, gap, between = 8 * scale, 5 * scale, 16 * scale
    entry_w = [legend_font.getlength(text) + swatch + gap for _, text in legend]
    legend_w = sum(entry_w) + between * max(len(legend) - 1, 0) if legend else 0
    legend_h = (round(13 * scale) + 8 * scale) if legend else 0
    image = Image.new('RGB', (round(max(table_w, legend_w)) + 2 * margin,
                              table_top + table_h + legend_h + margin), '#ffffff')
    draw = ImageDraw.Draw(image)
    draw.text((margin, margin + title_h / 2), title, font=title_font, fill='#111111', anchor='lm')
    draw.text((margin, margin + title_h + 2 * scale + subtitle_h / 2), subtitle, font=subtitle_font, fill='#444444', anchor='lm')
    # Grid color underneath, every cell painted inside it: 1px lines everywhere.
    draw.rectangle([margin, table_top, margin + table_w - 1, table_top + table_h - 1], fill='#dddddd')
    y = table_top + grid
    for row, height in zip(rows, heights):
        x = margin + grid
        for column, cell in enumerate(row):
            width = widths[column]
            draw.rectangle([x, y, x + width - 1, y + height - 1], fill=cell['fill'])
            middle = y + height / 2
            encoded = logo_data(cell['logo']) if cell['logo'] else None
            if encoded:
                mark = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGBA')
                mark.thumbnail((logo_px, logo_px), Image.LANCZOS)
                image.paste(mark, (round(x + (width - mark.width) / 2), round(middle - mark.height / 2)), mark)
            elif cell['text']:
                anchor, left = ('rm', x + width - padding(column)) if cell['right'] else ('mm', x + width / 2)
                draw.text((left, middle), cell['text'], font=cell['font'], fill=cell['color'], anchor=anchor,
                          stroke_width=scale // 2 if cell['stroke'] else 0, stroke_fill=cell['color'])
            x += width + grid
        y += height + grid
    # Confidence key under the table, one line, ending flush with its right
    # edge the way the page's own key does.
    right = margin + max(table_w, legend_w)
    y = table_top + table_h + 8 * scale
    x = right - legend_w
    for (color, text), width in zip(legend, entry_w):
        draw.rectangle([x, y + 2 * scale, x + swatch, y + 2 * scale + swatch],
                       fill=color, outline='#cccccc')
        draw.text((x + swatch + gap, y), text, font=legend_font, fill='#444444', anchor='la')
        x += width + between
    buffer = io.BytesIO()
    image.save(buffer, 'PNG', optimize=True, dpi=(72 * scale, 72 * scale))
    return buffer.getvalue(), image.width // scale


# Browser side of copy_picks_widget: the PNG is already on the page, so this
# only hands its bytes to the clipboard -- built synchronously inside the
# click (Safari refuses a clipboard write that starts after an await).
# Anything that stops it (no ClipboardItem, a refused write) falls through
# to the label's own checkbox and reveals the picture instead.
PICKS_COPY_SCRIPT = '''
function copyPicksImage(event){
 const toggle=document.getElementById("copy-picks-toggle");if(toggle.checked)return;
 if(!(window.ClipboardItem&&navigator.clipboard&&navigator.clipboard.write))return;
 event.preventDefault();
 const bytes=atob(document.getElementById("picks-png").src.split(",")[1]),data=new Uint8Array(bytes.length);
 for(let i=0;i<bytes.length;i++)data[i]=bytes.charCodeAt(i);
 navigator.clipboard.write([new ClipboardItem({"image/png":new Blob([data],{type:"image/png"})})]).then(()=>{
  const feedback=document.getElementById("copy-feedback");
  feedback.classList.remove("show");void feedback.offsetWidth;feedback.classList.add("show");
 },()=>{toggle.checked=true});
}
'''


# The headline sheet's columns, in order, and how each is drawn. Three
# places used to hardcode this independently -- headline_table's own
# select/relabel, the .headline-table-light nth-child CSS, and picks_png's
# index sets -- and they had already drifted out of step (picks_png
# right-aligned the home-logo column and missed the last figure column).
# One list now, and a test pins the CSS to it.
#
# Kickoff is three columns rather than one so the day, the date and the
# time each line up down the sheet instead of sliding around with the
# length of the one before it; the QB's rating is split off his name for
# the same reason. Neither split adds a header -- they read as one field.
# Mirrored about the spread: each side reads quarterback, rating, crest,
# team inwards towards the numbers, so the two teams sit either side of
# the market line instead of the sheet running left to right.
HEADLINE_COLUMNS = ['kick_day', 'kick_date', 'kick_time',
                    'away_qb', 'away_elo', 'away_logo', 'away_team',
                    'spread', 'prediction',
                    'home_team', 'home_logo', 'home_qb', 'home_elo',
                    'diff', 'pick', 'total', 'total_prediction', 'total_diff', 'total_pick']
HEADLINE_LABELS = ['Kickoff', '', 'ET', 'QB', 'Elo', '', 'Away', 'Spread', 'Model',
                   'Home', '', 'QB', 'Elo', 'Diff', 'Picks', 'O/U', 'Model', 'Diff', 'Picks']
# Figures line up on their last digit; everything else stays centred.
HEADLINE_RIGHT = {'kick_time', 'away_elo', 'home_elo', 'spread', 'prediction',
                  'diff', 'total', 'total_prediction', 'total_diff'}
# Drawn a size DOWN: the quarterback and his rating are there to be
# checked, not compared across rows like the figures are, and at the same
# size as everything else the names were the loudest thing on the sheet.
HEADLINE_SMALL = {'away_qb', 'home_qb', 'away_elo', 'home_elo'}
# Just wide enough for a 16px crest.
HEADLINE_TIGHT = {'away_logo', 'home_logo'}
# The kickoff day reads as a label, not a figure.
HEADLINE_LEFT = {'kick_day'}
# Team codes carry the row; the picks highlight lands on them.
HEADLINE_BOLD = {'away_team', 'home_team'}
# A rating tucks against the quarterback it belongs to.
HEADLINE_QB = {'away_qb', 'home_qb'}
# What a row says when no bucket covers it -- a dash, not a word, so the
# eye skips it and the picks are what stand out (see NO_PICK).
PASS_MARK = '–'


def headline_style():
    """The position-keyed half of the sheet's CSS, generated from
    HEADLINE_COLUMNS.

    Written by hand, these rules went stale every time the sheet was
    reordered, and silently: a block meant for "the two quarterback
    columns" kept styling positions 4 and 7 after those became the away
    crest and the away rating, which is how one rating ended up in Arial
    while the other stayed in the NFL font. Generating them means the
    column order is stated once and the stylesheet follows it.

    Both themes get identical geometry; only the rating's colour differs,
    because the page is dark and the picture is light."""
    def group(css_class, names, body):
        if not names:
            return ''
        selectors = ','.join(f'.{css_class} td:nth-child({i})' for i in headline_indices(names))
        return f'{selectors}{{{body}}}\n'

    rules = ''
    for css_class, rating_colour in (('headline-table', '#8b97a3'), ('headline-table-light', '#666')):
        # Figures line up on their last digit; the day reads as a label.
        rules += group(css_class, HEADLINE_RIGHT, 'text-align:right')
        rules += group(css_class, HEADLINE_LEFT, 'text-align:left')
        # A crest column is worth exactly one crest.
        rules += group(css_class, HEADLINE_TIGHT, 'padding:1px 2px;width:1%')
        # The quarterback and his rating read as one field, a size down.
        rules += group(css_class, HEADLINE_SMALL, 'font-size:10.5px')
        rules += group(css_class, HEADLINE_QB, 'padding-right:2px')
        rules += group(css_class, {'away_elo', 'home_elo'},
                       f'color:{rating_colour};padding-left:1px;font-family:Graduate,Georgia,serif')
        rules += group(css_class, HEADLINE_BOLD, 'font-weight:600')
        rules += group(css_class, {'kick_date'}, 'padding-left:2px;padding-right:3px')
    return rules


def headline_indices(names, base=1):
    """Which column positions a set of column names occupies."""
    return [i + base for i, column in enumerate(HEADLINE_COLUMNS) if column in names]


STYLE += headline_style()


def headline_table(folder, light=False, guide=True):
    """Per-game summary across both markets -- same mechanism and column
    order as main.py's original h_to_the_tml (pandas Styler, Greens/Reds
    background_gradient on diff/sd, #ffe590 yellow highlight only on cells
    that ARE the qualifying pick), plus O/U appended in the same flat style
    with compact team labels. Spread/Model are shown
    from the away team's own perspective (negative = away favored), same
    convention as the per-game card's header. Built from whichever
    {market}_details.csv this folder already has (each
    write_packets(..., market=...) call saves its own). 'Picks' is a real
    qualify/pass call from HIGH_CONFIDENCE_CUTOFFS, not "always show a
    lean" -- most games should say PASS, unhighlighted.

    light=True builds the same table with a white-background/dark-text
    theme (.headline-table-light) and pastel gradients instead of the
    on-page dark ones -- meant for the "Copy" button's clipboard payload,
    which needs to look right pasted into an email/Slack/doc, not for
    display in this dark packet."""
    from backtester import settle
    frames = {}
    for market in ['spread', 'total']:
        path = folder / f'{market}_details.csv'
        if path.exists():
            cutoffs = HIGH_CONFIDENCE_CUTOFFS[market]
            frames[market] = apply_tiers(settle(pd.read_csv(path), cutoffs['diff_cutoff'],
                                                cutoffs['sd_cutoff']), market, running_model(folder, market))
    if not frames:
        return ''
    base = next(iter(frames.values()))
    sched = packet_schedule()[['season', 'week', 'away_team', 'home_team', 'gameday', 'gametime']]
    games = base[['season', 'week', 'away_team', 'home_team']].merge(
        sched, on=['season', 'week', 'away_team', 'home_team'], how='left')

    def quarterback(row, side):
        """The scheduled starter's name, and his rating going in, as two
        values -- they share a heading on the sheet but sit in their own
        columns so the ratings line up down the page instead of trailing
        off the end of names of different lengths."""
        name = row.get(f'{side}_qb_short')
        if pd.isna(name) or not str(name).strip():
            name = row.get(f'{side}_qb_name')
        if pd.isna(name) or not str(name).strip():
            return '', ''
        elo = row.get(f'{side}_raw_off_qb_elo')
        return str(name), ('' if pd.isna(elo) else f'{elo:.1f}')

    def market_fields(frame, g, market):
        match = frame[(frame.away_team == g.away_team) & (frame.home_team == g.home_team)] if frame is not None else None
        if match is None or match.empty:
            return dict(line=np.nan, model=np.nan, diff=np.nan, sd=np.nan, tier=None, pick='',
                    away_qb='', away_elo='', home_qb='', home_elo=''), None
        r = match.iloc[0]
        pick = (r.away_team if r.edge > 0 else r.home_team) if market == 'spread' else ('OVER' if r.edge > 0 else 'UNDER')
        pick = pick if bool(r.qualifies) else PASS_MARK
        # market_base/prediction are stored away-minus-home; flip sign for
        # spread so both read as the away team's own line ("team -X" =
        # away favored by X), matching game_header's Market/Model rows.
        # Total has no team-perspective concept, so it's left alone.
        sign = -1 if market == 'spread' else 1
        (away_qb, away_elo), (home_qb, home_elo) = quarterback(r, 'away'), quarterback(r, 'home')
        return (dict(line=sign * r.market_base, model=sign * r.prediction, diff=abs(r.edge), sd=r.sd,
                     market_line=r.market_base, tier=r.get('tier'), pick=pick,
                     away_qb=away_qb, away_elo=away_elo, home_qb=home_qb, home_elo=home_elo),
                (pick if pick != PASS_MARK else None))

    rows = []
    picks = set()
    for _, g in games.iterrows():
        spread, spread_pick = market_fields(frames.get('spread'), g, 'spread')
        total, total_pick = market_fields(frames.get('total'), g, 'total')
        for pick in [spread_pick, total_pick]:
            if pick:
                picks.add(pick)
        # Same shape/column names as h_to_the_tml (minus qb/qb_elo): away,
        # then spread's own line/prediction, then home, then diff/sd/pick --
        # O/U's own total/total_prediction/diff/sd/pick block appended
        # after, team names not repeated for the second market.
        # The spread frame carries the quarterbacks; the total frame is the
        # same games, so either will do -- prefer whichever answered.
        qbs = spread if spread.get('away_qb') else total
        rows.append(dict(
            gameday=g.gameday, gametime=g.gametime,
            away_logo=g.away_team, away_team=g.away_team, away_qb=qbs.get('away_qb', ''),
            away_elo=qbs.get('away_elo', ''), home_elo=qbs.get('home_elo', ''),
            spread=spread['line'], prediction=spread['model'],
            home_qb=qbs.get('home_qb', ''), home_team=g.home_team, home_logo=g.home_team,
            diff=spread['diff'], pick=spread['pick'],
            total=total['line'], total_prediction=total['model'],
            total_diff=total['diff'], total_pick=total['pick'],
            spread_tier=(spread['tier'] or '') if spread_pick else '',
            total_tier=(total['tier'] or '') if total_pick else ''))
    table = pd.DataFrame(rows)
    table['gameday'] = pd.to_datetime(table.gameday)
    table['gametime'] = pd.to_datetime(table.gametime, format='%H:%M', errors='coerce').dt.time
    table = table.sort_values(['gameday', 'gametime', 'away_team']).reset_index(drop=True)
    # Tiers ride along for the sort, then step out of the displayed frame.
    tiers = table[['spread_tier', 'total_tier']]
    table = table.drop(columns=['spread_tier', 'total_tier'])
    # Day, date and time as three columns: as one string the pieces slid
    # around with each other's length ("Thu 09/24 8:15" against
    # "Sun 09/27 12:00"), so nothing lined up down the sheet. Split, each
    # sits in its own plane and the column is no wider for it. Times carry
    # am/pm -- every kickoff is Eastern, which the heading says once.
    table['kick_day'] = [day.strftime('%a') for day in table.gameday]
    table['kick_date'] = [day.strftime('%m/%d') for day in table.gameday]
    table['kick_time'] = [time.strftime('%I:%M %p').lstrip('0').lower() if time else ''
                          for time in table.gametime]
    table = table.drop(columns=['gameday', 'gametime'])
    table = table[HEADLINE_COLUMNS]

    def signed(value, precision=1):
        return '—' if pd.isna(value) else f'{value:+.{precision}f}'

    def plain(value, precision=1):
        return '—' if pd.isna(value) else f'{value:.{precision}f}'

    def tier_styles(frame):
        """Colour each pick cell (and the picked team) by its confidence tier."""
        styles = pd.DataFrame('', index=frame.index, columns=frame.columns)
        for i, row in frame.iterrows():
            for column, tier in [('pick', tiers.spread_tier[i]), ('total_pick', tiers.total_tier[i])]:
                if not tier:
                    continue
                paint = f'background-color: {TIER_COLORS[tier]}; color: #222; font-weight: 600'
                styles.loc[i, column] = paint
                if column == 'pick' and row['pick'] in (row.away_team, row.home_team):
                    styles.loc[i, 'away_team' if row['pick'] == row.away_team else 'home_team'] = paint
        return styles

    def team_cell(code):
        return escape(code)

    from matplotlib.colors import LinearSegmentedColormap
    # The size of a disagreement is shaded blue, not green: green is what a
    # qualifying A pick is painted (TIER_COLORS), and a deep green Diff cell
    # beside it read as though the row had been picked when it had not.
    # Blue is unused by any tier, so the two can never be mistaken.
    if light:
        blues = LinearSegmentedColormap.from_list('headline_blues_light', ['#eaf1fa', '#4a86c8'])
        reds = LinearSegmentedColormap.from_list('headline_reds_light', ['#fdf0ef', '#e2726b'])
    else:
        blues = LinearSegmentedColormap.from_list('headline_blues', ['#141a21', '#1d4368'])
        reds = LinearSegmentedColormap.from_list('headline_reds', ['#1b1717', '#643536'])

    # Column order is fixed and documented here (also relied on by the
    # .headline-table nth-child CSS that right-aligns the numeric columns):
    # gameday/gametime/away_logo/away_team/spread/prediction/home_team/
    # home_logo/diff/pick, then O/U's total/total_prediction/total_diff/
    # total_pick. No SD column in either market -- the pick's tier colour
    # (PICK_BUCKETS) is the confidence signal now. relabel_index
    # only changes the displayed header text -- .format()/.map() below
    # still key off the real column names. background_gradient/
    # highlight_picks need per-cell inline styles since they're data-dependent.
    # write_packets() calls this once per market (spread, then total) -- on
    # the first (spread) call, total_details.csv doesn't exist yet, so
    # total_diff is entirely NaN for that pass (harmlessly
    # overwritten once the second call has both files). background_gradient
    # over an all-NaN column makes pandas call np.nanmin/nanmax on nothing,
    # which is a real (if harmless) numpy RuntimeWarning -- skip gradients
    # for whichever of these columns are actually all-NaN instead of
    # letting that warning fire every single run.
    gradient_diff = [c for c in ['diff', 'total_diff'] if table[c].notna().any()]
    gradient_sd = []  # SD is no longer a column: the tier colour carries confidence
    css_class = 'headline-table-light' if light else 'headline-table'
    styled = table.style.hide(axis='index').set_table_attributes(f'class="{css_class}"')
    if gradient_diff:
        styled = styled.background_gradient(subset=gradient_diff, cmap=blues)
    if gradient_sd:
        styled = styled.background_gradient(subset=gradient_sd, cmap=reds)
    styled = (styled.format({
                 'spread': signed, 'prediction': signed, 'diff': plain,
                 'total': plain, 'total_prediction': plain, 'total_diff': plain,
                 'away_logo': lambda x: logo(x), 'home_logo': lambda x: logo(x),
                 'away_team': team_cell, 'home_team': team_cell,
             })
             .relabel_index(HEADLINE_LABELS, axis=1)
             .apply(tier_styles, axis=None))
    behind = {market: running_model(folder, market) for market in ['spread', 'total']}
    if not guide:
        # For the picture (picks_png): the sheet and who produced it, nothing
        # else. picks_png flattens every table it is handed into one grid, so
        # the guide's bucket notes -- full sentences on a single unwrapped
        # line -- would set the image's width, which is how the picks table
        # ended up drawn 6983 CSS px wide with the columns pushed to either
        # edge. The tier key is redundant there too: picks_png draws its own
        # from `legend`.
        return styled.to_html()
    return styled.to_html() + tier_legend()


def write_packets(predictions, panel, importance, config, root):
    root = Path(root)
    market = config['market']
    for (season, week), games in predictions.groupby(['season', 'week']):
        shared = config['calculation'] in ['shared-scoring-v1', 'joint-matchup-v1', 'two-sided-team-points-v1', model_spec.ID]
        snapshot = display_stats(season, week, config['lookback']) if market == 'spread' or shared else pd.DataFrame()
        # The Stats tab's "Model" view uses the recency preset the model was
        # actually built with (config['feature_calculation'], set by
        # two_sided_packet; 'steep' for runs from before it was recorded).
        # Scoped to this model family rather than guessed at for others.
        weighted_snapshot = (display_stats(season, week, config['lookback'],
                                           calculation=config.get('feature_calculation', 'steep'))
                             if config['calculation'] in ('two-sided-team-points-v1', model_spec.ID)
                             and not snapshot.empty else None)
        folder = root / f'{int(season)}_{int(week):02d}'
        folder.mkdir(parents=True, exist_ok=True)
        # Cards in kickoff order, not whatever order the panel happened to
        # be in (alphabetical by away_team from build_panel's own sort).
        # Defensive: not every caller's frame has season/away_team/home_team
        # in a shape packet_schedule can join on (e.g. synthetic test fixtures) --
        # leave the existing order alone rather than erroring in that case.
        if 'gameday' in games and {'season', 'week', 'away_team', 'home_team'}.issubset(games.columns):
            kickoff = packet_schedule()[['season', 'week', 'away_team', 'home_team', 'gametime']]
            games = games.merge(kickoff, on=['season', 'week', 'away_team', 'home_team'], how='left', suffixes=('', '_sched'))
            games = games.sort_values(['gameday', 'gametime', 'away_team'])
        title = f'{int(season)} · Week {int(week)} · {"Against the spread" if market == "spread" else "Over / under"}'
        header = packet_tabs(market) + f'<h1>{title}</h1>'
        from backtester import settle
        cutoffs = HIGH_CONFIDENCE_CUTOFFS[market]
        # config knows which model produced these predictions; the folder does
        # not yet -- {market}_config.json is written further down, after the
        # pages are rendered. Reading the folder here silently fell back to
        # the default model and threw away every shared-model pick.
        running = config.get('tier_model') or running_model(folder, market)
        games = apply_tiers(settle(games, cutoffs['diff_cutoff'], cutoffs['sd_cutoff']), market, running)
        cards = []
        for _, row in games.iterrows():
            lean = (row.away_team if row.edge > 0 else row.home_team) if market == 'spread' else ('OVER' if row.edge > 0 else 'UNDER')
            qualifies = bool(row.qualifies)
            action = lean if qualifies else 'PASS'
            stats = panel[(panel.season == season) & (panel.week == week) & (panel.away_team == row.away_team) & (panel.home_team == row.home_team)]
            chart = matchup_attribution(row, snapshot, stats, shared=shared,
                    differential=config.get('input_mode') == 'differential') if market == 'spread' or shared else attribution(row)
            if config['calculation'] == 'shared-scoring-v1' and config.get('attribution_schema', 0) < 2:
                chart = '<p class="warn">Feature labels in this saved preview need regeneration after an attribution-column correction. Scores remain usable as experimental predictions. Rerun shared_scoring.py for corrected explanations.</p>'
            # The Calculation-notes disclosure is identical boilerplate on
            # every single card on this page (market/shared/differential are
            # page-wide, not per-game) -- said once, generically, in the
            # footer instead (below) rather than repeated per matchup.
            chart = re.sub(r'<details><summary>Calculation notes</summary>.*?</details>', '', chart, flags=re.DOTALL)
            cards.append(f'<section class="card">{game_header(row, market, action)}{chart}</section>')
        # Notes: short, plain and accurate, closed by default at the bottom
        # of the page. The attribution notes describe two_sided_packet's
        # shared network (see fit_two_sided and matchup_attribution); other
        # model families get the generic version.
        spread_page = market == 'spread'
        behind_page = running
        travelling = 'attr_away_travel_adv' in games.columns
        notes = [
            # An SD condition is optional per market; neither uses one now.
            ('Picks', 'A pick needs an edge of at least '
                      + (f'{cutoffs["diff_cutoff"]:g} points'
                         + (f' and an SD of at most {cutoffs["sd_cutoff"]:.2f}' if cutoffs.get('sd_cutoff') else ''))
                      + ', and it has to land in a bucket the backtest measured above 54%: '
                      + '; '.join(f'{tier_band(b["rate"])} {b["rule"]} ({100 * b["rate"]:.1f}%)'
                                  for b in sorted(PICK_BUCKETS[market]['buckets'], key=lambda b: -b['rate']))
                      + f'. Anything else shows – and is not a bet: {NO_PICK[market]["record"]}, no era above '
                        'break-even. The letter is the bucket’s measured rate (S 60%+, A 57.5-60%, B 54-57.5%), '
                        'not a judgement about this game. Break-even is 52.4%. Lines are stored market lines, '
                        'not live odds, and starting QBs aren’t verified.'),
            ('Numbers', f'{"ModelLine" if spread_page else "Model O/U"} and the predicted scores are averages across the '
                        'model’s ensemble; SD is how much its runs disagree, not game risk. QB Elo is the scheduled '
                        f'starter’s recency-weighted rating going into the game. Stat ranks use pregame rates over the previous {config["lookback"]} '
                        'regular-season weeks; #1 is best.'),
        ]
        if config.get('context_note'):
            notes.append(('Context data', config['context_note']))
        if shared:
            notes += [
                ('How the bars work', 'Each bar is one input’s share of this game’s prediction, measured from an average '
                                      'training game (every input at its training average) and averaged across the ensemble. '
                                      + ('Bars are in away-line terms: negative favors the away team, positive the home team.'
                                         if spread_page else 'Bars are in points: positive raises the total.')),
                ('Offense and defense bars', 'Both teams are scored by the same network. The first bar '
                                             + ('nets' if spread_page else 'adds') + ' each team’s offense-vs-opposing-defense '
                                             'inputs to its own score, so it also reflects the home offense against the away '
                                             'defense, not only the matchup in its label. The second bar does the same for each '
                                             'team’s defense-vs-opposing-offense inputs. Expand either bar for one row per stat.'),
                # Travel is Model 2.2's input; older versions' packets keep the
                # two-term wording rather than describing a bar they don't have.
                (('Home field, rest and travel' if travelling else 'Home field and rest'),
                 'Fixed adjustments: a learned weight times the home-field'
                 + (', rest-day or travel-distance difference, measured from a neutral site with equal rest and equal '
                    'travel. Travel is the flight from each team’s home stadium to this one, so the home team is '
                    'normally 0 and only the difference between the two matters.' if travelling else
                    ' or rest-day difference, measured from a neutral site with equal rest.')),
                ('Weather', 'The model uses feels-like temperature, wind and precipitation. Indoor games get fixed '
                            '72°F, calm, dry readings. The bars are measured from the average training game, so an '
                            'ordinary day can still show small ones; those come from that average, not the conditions. '
                            + ('Both teams share the same weather, so it only moves the spread through how it combines '
                               'with each team’s stats, and those effects vary a lot between the ensemble’s runs. '
                               if spread_page else '')
                            + 'A retractable roof the schedule doesn’t list yet is treated as outdoors.'),
            ]
        else:
            notes.append(('How the bars work', 'Each bar is one input’s share of this game’s prediction relative to the '
                                               f'model’s baseline. {config.get("baseline_note", "")}'.strip()))
        notes.append(('Caveat', 'Bars explain the model’s prediction, not what causes games to turn out a certain way.'))
        # Full set, not just the top 12 -- scrollable container keeps the
        # page from turning into one giant bar chart while still letting
        # you page through every feature, not just the headline few.
        table = importance.copy()
        table['feature'] = table.feature.map(pretty)
        note = config.get('importance_note', 'Each bar is a paired refit/drop test: remove one feature, retrain, '
                          'and see how prediction error on the training sample changed. Positive means removing '
                          'that feature made the model worse (it was pulling weight); negative means removing it '
                          'made the model better (it was actively hurting predictions). This is a discovery '
                          'diagnostic on the training sample, not held-out betting evidence, and because many '
                          'features are correlated with each other, one can substitute for another -- a low score '
                          "doesn't mean a feature is useless, just that something else already covers it.")
        fi = (f'<section class="card"><h2>Feature importance</h2><details><summary>Method</summary>{escape(note)}</details>'
             + f'<div class="importance-scroll">{importance_bars(table)}</div></section>')
        if config['calculation'] == 'shared-scoring-v1' and config.get('attribution_schema', 0) < 2:
            fi = '<p class="warn">Saved feature-importance labels also require regeneration. They are hidden until the shared-scoring preview is rerun.</p>'
        games.to_csv(folder / f'{market}_details.csv', index=False)
        importance.to_csv(folder / f'{market}_importance.csv', index=False)
        (folder / f'{market}_config.json').write_text(json.dumps({k: v for k, v in config.items() if k != 'spec'}, indent=2),
                                                     encoding='utf-8')
        footer = ('<details class="sheet-notes"><summary>Notes</summary>'
                  + ''.join(f'<p><strong>{escape(title)}.</strong> {escape(text)}</p>' for title, text in notes) + '</details>')
        (folder / f'{market}.html').write_text(page(title, header + ''.join(cards) + footer, 'packet'), encoding='utf-8')
        (folder / f'{market}_importance.html').write_text(fi, encoding='utf-8')
        evidence = ''.join(f'<h2>{name.title()}</h2>' + (folder / f'{name}_importance.html').read_text()
                           for name in ['spread', 'total'] if (folder / f'{name}_importance.html').exists())
        (folder / 'importance.html').write_text(page(title, packet_tabs('importance') + evidence, 'packet'), encoding='utf-8')
        headline = (f'<h1>Model</h1><p class="report-date">{int(season)} · Week {int(week)}</p>'
                   + copy_picks_widget(season, week, headline_table(folder, light=True, guide=False))
                   + headline_table(folder))
        (folder / 'index.html').write_text(
            page(title, packet_tabs('headline') + headline, 'packet headline-shell sheet-page'), encoding='utf-8')
        (folder / 'stats.html').write_text(page(title, packet_tabs('stats') + '<h1>Stats</h1>'
            + stats_tables(snapshot, games, season, week, config['lookback'], weighted_snapshot), 'packet headline-shell'), encoding='utf-8')
        if config.get('spec'):
            (folder / 'specs.html').write_text(
                page(title, packet_tabs('specs')
                     + specs_page(config['spec'],
                                  {m: running_model(folder, m) for m in ['spread', 'total']}), 'packet'),
                                               encoding='utf-8')
        for missing in ['spread', 'total']:
            if not (folder / f'{missing}.html').exists():
                (folder / f'{missing}.html').write_text(page(title, packet_tabs(missing) +
                    '<p>This model has no saved analysis for this market yet. Regenerate the preview to add it.</p>', 'packet'), encoding='utf-8')
        bundle_single_file(folder)


def write_research_report(summaries, grid, output):
    from optimize_picks import policy_sd_cutoff
    cards = []
    for market, config in summaries.items():
        cards.append(f'<section class="card"><h2>{market.title()} · {config["status"]}</h2><p>{escape(config["reason"])}</p>')
        if 'validation' in config:
            cards.append(pd.DataFrame([config['calibration'], config['validation']], index=['Calibration (selected)', 'Validation (fixed)']).to_html(float_format=lambda x: f'{x:.3f}', border=0))
            sd = policy_sd_cutoff(config)
            cards.append(f'<p>Minimum edge: {config["diff_cutoff"]} points. Maximum SD: {sd if sd is not None else "none"} points. Validation ROI 95% interval: {config["validation_roi_95"]}.</p><p>Features: {escape(", ".join(map(pretty, config["features"])))}</p>')
        cards.append('</section>')
    neural = output / 'neural_summary.json'
    if neural.exists():
        cards.append('<section class="card"><h2>Actual neural confirmation</h2><p class="muted">Separate model-specific calibration; these are not the unchanged 100-member production ensemble.</p>')
        rows = []
        for market, config in json.loads(neural.read_text()).items():
            if 'validation' in config:
                rows.append(dict(market=market, model=config['model'], status=config['status'],
                                 edge_cutoff=config['diff_cutoff'], sd_cutoff=policy_sd_cutoff(config),
                                 **config['validation']))
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table[['market', 'model', 'status', 'n', 'win_rate', 'pnl_units', 'roi']]
        cards.append(table.to_html(index=False, float_format=lambda x: f'{x:.3f}', border=0))
        cards.append('<p><a href="neural_summary.json">Neural configuration and ROI intervals</a></p></section>')
    links = ''.join(f'<li><a href="weeks/{p.parent.name}/index.html">{p.parent.name.replace("_", " · Week ")}</a></li>' for p in sorted((output / 'weeks').glob('*/index.html')))
    header = '<div class="eyebrow">NFL / Model research</div><h1>Spread & total evidence</h1><p class="warn">Retrospective validation, not guaranteed future profit. These seasons have already been explored. Keep production unchanged; paper-test any challenger prospectively. Ridge variance cutoffs do not apply to the neural ensemble.</p><p>All fits use earlier weeks only. Discovery ranks features; calibration selects feature/calculation/cutoff combinations; the final period is evaluated after choices are fixed. Prices fall back to −110 only when missing. Exact zero edges are passes; pushes return zero. Flat one-unit risk, no Kelly sizing.</p>'
    (output / 'report.html').write_text(page('NFL research', header + ''.join(cards) + f'<h2>Weekly packets</h2><ul>{links}</ul><p><a href="cutoff_grid.csv">Full calibration grid</a> · <a href="summary.json">Model-specific configuration</a></p>'), encoding='utf-8')


def neural_packet(season, week, lookback=20, iterations=100, seed=1337, symmetric=False):
    """The existing ensemble, both markets; never borrow a Ridge bet cutoff."""
    import model_shredski as ms
    import optimize_picks as op
    import utils
    panel = op.build_panel(season, week, lookback, lookback, 'mean')
    if symmetric:
        panel = op.symmetric_features(panel)
    calculation = 'mean-symmetric-v1' if symmetric else 'mean'
    output = Path(f'data/results/{season}_{week}_{lookback}/{"packet_symmetric" if symmetric else "packet"}')
    headline = None
    for market in (['spread'] if symmetric else ['spread', 'total']):
        features = [f for f in op.feature_names(panel, market) if f != 'away_game_importance']
        data = op.market_panel(panel, market)
        target = data[(data.season == season) & (data.week == week)]
        if target.empty:
            print(f'{market}: no stored market line; packet skipped')
            continue
        fingerprint = pd.util.hash_pandas_object(panel[op.KEY + features + ['away_score', 'home_score']], index=False).values.tobytes().hex()
        cached = utils.cache_path('neural_predictions', [fingerprint, market, features, iterations, seed, 100],
                                  ['model_shredski.py', 'modelo_workers.py'])
        artifacts = cached.with_suffix('')
        if not cached.exists():
            ms.modelo(panel, season, week, artifacts, bt=False, features=features, market=market,
                      iterations=iterations, random_state=seed, round_predictions=False)
            utils.save_parquet(pd.read_csv(artifacts / 'explanations.csv'), cached)
        details = pd.read_parquet(cached)
        if market == 'spread':
            headline = details[['away_team', 'home_team', 'prediction', 'variance']].copy()
            headline.attrs['model_spec'] = op.neural_spec(features, lookback, calculation, iterations, seed)
            headline.attrs['model_spec']['train_weeks'] = lookback
        predictions = target.merge(details, on=['away_team', 'home_team'], validate='one_to_one')
        predictions['edge'] = predictions.prediction - predictions.market_base
        config = dict(model=f'production-architecture-neural / {iterations} members / seed {seed}',
                      market=market, lookback=lookback, calculation=calculation, status='PASS',
                      headline_href=f'../../html_{season}_{week}_{lookback}.html',
                      reason='Neural-specific profitable cutoffs not established',
                      baseline_note='Baseline is the ensemble prediction at mean training features.',
                      importance_note='Training-sample permutation diagnostic, not held-out predictive evidence. Error bars in the CSV measure variation across ensemble members, not confidence in betting profitability.')
        importance = pd.read_csv(artifacts / 'importance.csv').sort_values('importance', ascending=False)
        write_packets(op.settle(predictions), panel, importance, config, output)
    print(f'Weekly packet: {output / f"{season}_{week:02d}" / "index.html"}')
    return headline


def refresh_weather(season, week, forecast_file=None):
    """Observed weather for games played since the last pull, then a fresh
    forecast for the target week (bypassing pull_weather's hourly live cache).
    Best effort on purpose: two_sided_packet already falls back to a pulled
    forecast when a game has no reanalysis row -- and pulls one itself if it
    has to -- so a weather hiccup here should not sink the whole rerun."""
    from types import SimpleNamespace
    import pull_weather
    try:
        print(f'Refresh: topping up observed weather for {season}...', flush=True)
        _, failures = pull_weather.pull_historical([season])
        if not failures.empty:
            print(failures.to_string(index=False), flush=True)
        pull_weather.build_weather_features()
    except Exception as error:
        print(f'Refresh: observed-weather top-up failed ({error}); keeping the existing '
              'data/weather/historical_features.parquet.', flush=True)
    output = Path(forecast_file or 'data/weather/forecasts.parquet')
    try:
        print(f'Refresh: pulling a live forecast for {season} wk{week}...', flush=True)
        games = pull_weather.scheduled_games(season, week, 'live', decision_hours=24)
        pull_weather.pull(games, SimpleNamespace(mode='live', decision_hours=24, publication_hours=8,
                                                 output=str(output), cache_dir='data/cache/open_meteo',
                                                 refresh=True))
    except Exception as error:
        print(f'Refresh: no live forecast pulled for {season} wk{week} ({error}); '
              f'using whatever {output} already holds.', flush=True)


def refresh_inputs(season, week, shared=True, forecast_file=None):
    """Trust nothing on disk: repull the source data and drop every derived
    cache, so the caller's fit recomputes the whole chain (weekly features,
    packet stats, model members) from scratch instead of reading data/cache.

    Repulls the target season's schedule and play-by-play, plus any earlier
    season still missing a score -- that is staleness, not a game in progress.
    Earlier seasons are final, so they are not re-downloaded wholesale; call
    data_pullson.pull_sched/pull_pbp directly for the rare backfill that needs it."""
    import data_pullson
    import utils
    utils.bypass_caches()
    print('Refresh: ignoring every cached feature, stat and model fit from before this run.', flush=True)
    seasons = {season}
    schedule = Path('data/sched.parquet')
    if schedule.exists():
        sched = pd.read_parquet(schedule)
        prior = (sched.season < season) | ((sched.season == season) & (sched.week < week))
        seasons |= set(sched.loc[prior & sched.away_score.isna(), 'season'].astype(int).tolist())
    seasons = sorted(seasons)
    print(f'Refresh: pulling schedule + play-by-play for {seasons}...', flush=True)
    data_pullson.pull_sched(seasons)
    data_pullson.pull_pbp(seasons)
    if shared:
        refresh_weather(season, week, forecast_file)


def refresh_packet(season, week, lookback=20, symmetric=False, shared=False):
    """Render saved forecasts only (the --restyle path): no model fits, data
    pulls, or headline edits. Needs {market}_details.csv/{market}_importance.csv
    on disk -- two_sided_packet (shared=True) no longer keeps those around (only
    packet.html + config.json, to avoid cluttering data/results/), so this has
    nothing to restyle for a two-sided packet produced since that change; refit
    it with --refresh instead."""
    if shared:
        root = model_spec.run_folder(season, week, lookback)
        folder = root
    else:
        root = Path(f'data/results/{season}_{week}_{lookback}/{"packet_symmetric" if symmetric else "packet"}')
        folder = root / f'{season}_{week:02d}'
    refreshed = False
    for market in ['spread', 'total']:
        details = folder / f'{market}_details.csv'
        if not details.exists():
            continue
        predictions = pd.read_csv(details)
        saved_config = folder / f'{market}_config.json'
        config = json.loads(saved_config.read_text()) if saved_config.exists() else dict(
            model='Production ensemble', market=market, lookback=lookback,
            calculation='mean', status='PASS', reason='Validated cutoffs not established',
            headline_href=f'../../html_{season}_{week}_{lookback}.html',
            baseline_note='Baseline: ensemble prediction at mean training features.',
            importance_note='Training-sample permutation diagnostic; not held-out feature evidence.')
        importance = pd.read_csv(folder / f'{market}_importance.csv')
        write_packets(predictions, predictions, importance, config, root)
        refreshed = True
    if not refreshed:
        raise ValueError(f'Nothing to restyle in {folder} -- no {{market}}_details.csv found. '
                         'Refit the model with --refresh instead.')
    print(f'Packet restyled: {folder / "packet.html"}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=f'Build a weekly packet by fitting the chosen model -- '
                                     f'{model_spec.LABEL} writes to {model_spec.RESULTS}/{{season}}_{{week}}_{{lookback}}/ '
                                     'with a model.json spec.')
    parser.add_argument('--model', choices=['model', 'two-sided', 'neural'], default='model',
                        help=f"'model' (default): {model_spec.LABEL} ('two-sided' is the same thing, its old name). "
                             "'neural': the original single-network packet (main.py --packet's model).")
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--week', type=int, required=True)
    parser.add_argument('--refresh', action='store_true',
                        help='Full rerun: repull schedule/play-by-play/weather, ignore every cached feature, '
                             'stat and model fit, and refit from scratch')
    parser.add_argument('--restyle', action='store_true',
                        help='The opposite of --refresh: re-render saved predictions without pulling or fitting '
                             'anything (neural packets only -- two-sided packets no longer keep the CSVs it needs)')
    # None means "whatever the model profile says" (model_spec.PROFILES).
    # These are model properties: the tier rates were measured at a specific
    # lookback and training window, so overriding them detaches the sheet's
    # numbers from what is being bet. The packet says so when you do.
    parser.add_argument('--lookback', type=int, default=None,
                        help="Override the model profile's stat lookback (default: the profile's own)")
    parser.add_argument('--train-window', type=int, default=None,
                        help="Override the model profile's training REG weeks (default: the profile's own)")
    parser.add_argument('--model-version', choices=list(f'model_{v}' for v in model_spec.VERSIONS), default='model_2.0',
                        help='Which model version to build: model_2.0 (default), model_2.1 (adds the EPA inputs) '
                             'or model_2.2 (adds the travel-distance input). Each writes to its own '
                             'data/results/model_*/ folder.')
    parser.add_argument('--calculation', default=None,
                        help="Stat recency preset (data_crunchski_2.DECAY_PRESETS): 'weighted' (default), "
                             "'steep', or 'carryover' (weighted, with earlier-season games counted half)")
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--weather-file', help='Two-sided only: historical/reanalysis weather')
    parser.add_argument('--forecast-file', help='Two-sided only: fallback for a target week with no historical '
                        'weather yet (i.e. not played yet) -- pull it first with pull_weather.py --season ... '
                        '--week ... --mode live. Defaults to data/weather/forecasts.parquet if it exists.')
    args = parser.parse_args()
    model_spec.select(args.model_version)   # also swaps the input set (2.1 adds EPA, 2.2 adds travel)
    if args.restyle:
        if args.refresh:
            parser.error('--restyle re-renders what is already saved; --refresh refits from new data. Pick one.')
        refresh_packet(args.season, args.week, args.lookback or model_spec.profile()['lookback'],
                       shared=args.model != 'neural')
    else:
        if args.refresh:
            refresh_inputs(args.season, args.week, shared=args.model != 'neural',
                           forecast_file=args.forecast_file)
        if args.model != 'neural':
            from shared_scoring import two_sided_packet
            two_sided_packet(args.season, args.week, args.lookback, args.train_window, args.iterations,
                             args.epochs, args.seed, args.jobs, args.weather_file, args.forecast_file,
                             calc=args.calculation)
        else:
            neural_packet(args.season, args.week, args.lookback or model_spec.profile()['lookback'],
                          args.iterations, args.seed)
