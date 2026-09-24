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
body{font:15px/1.5 system-ui,sans-serif;color:#203039;background:#f1f3f1;margin:0}
main{max-width:1050px;margin:auto;padding:40px 24px}h1,h2,h3{font-weight:600;line-height:1.2}
h1{font-size:34px;margin:8px 0}h2{font-size:24px}h3{font-size:17px}.eyebrow{letter-spacing:2px;text-transform:uppercase;font-size:12px;color:#64796f}
.muted,small{color:#63716d}.card{background:white;border:1px solid #dce3dd;border-radius:10px;padding:25px;margin:22px 0;break-inside:avoid}
.teams{display:flex;align-items:center;gap:15px}.logo{width:52px;height:52px;object-fit:contain}.metrics{display:flex;flex-wrap:wrap;gap:30px;margin:20px 0}.metric strong{display:block;font-size:25px;font-weight:550}
.pill{display:inline-block;background:#edf1ec;border-radius:4px;padding:4px 9px;font-size:12px;font-weight:600}.warn{border-left:3px solid #b18b4f;padding:10px 16px;background:#faf7f0}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px 10px;text-align:right;border-bottom:1px solid #edf0ed}td:first-child,th:first-child{text-align:left}a{color:#236950}
.columns{display:grid;grid-template-columns:1.2fr 1fr;gap:28px}.barrow{display:grid;grid-template-columns:180px 1fr 55px;align-items:center;gap:8px;font-size:12px;margin:7px 0}.track{position:relative;height:12px;background:linear-gradient(90deg,#f7f4ef 50%,#eef4f0 50%)}.bar{position:absolute;height:12px;background:#30765d}.negative{background:#b17f51}.value{text-align:right;font-variant-numeric:tabular-nums}
/* Full feature list, not just the top 12 -- scrolls in place instead of
   turning the whole page into one long bar chart. */
.importance-scroll{max-height:480px;overflow-y:auto;padding-right:6px;border:1px solid #ffffff12;border-radius:6px}
.importance-scroll .barrow{margin:7px 10px}
@media(max-width:760px){.columns{grid-template-columns:1fr}.barrow{grid-template-columns:145px 1fr 45px}main{padding:20px 12px}}
@media print{body{background:white}main{padding:0}.card{border-radius:0;page-break-inside:avoid}a{color:inherit}.no-print{display:none}}
'''

STYLE += '''
body{font-family:Arial,sans-serif;background:white;color:#222}
main{max-width:1150px;padding:24px}h1{font-size:26px}h2{font-size:21px}
.card{border:2px solid #aaa;border-radius:0;padding:20px;margin:20px 0}
td,th{border:1px solid #bbb;padding:9px 12px}th{background:#eee}
.pill{border-radius:0}.eyebrow{letter-spacing:0}.columns{display:block}
.rank{color:#666;font-size:12px;margin-left:8px}.contribution{position:relative;height:26px;background:linear-gradient(90deg,#f5eee6 50%,#edf4ef 50%)}
.contribution:after{content:"";position:absolute;left:50%;height:100%;border-left:1px solid #aaa}
.contribution .bar{height:26px;opacity:.45}.contribution b{position:relative;z-index:1;display:block;text-align:center;line-height:26px;font-size:12px}
.reconcile{border-top:2px solid #999;padding-top:10px;text-align:right}
details{margin:12px 0;font-size:13px;color:#666}summary{cursor:pointer}
.packet{max-width:800px;background:#111;color:#eee}
body:has(.packet){background:#111}
.packet .card{border:0;border-top:1px solid #353535;background:transparent;padding:24px 0}
.packet .muted,.packet small,.packet details,.packet .rank{color:#aaa}
.packet a{color:#8bb9ff}.packet .pill{background:#292929;color:#ccc}
.packet .warn{background:#28231b;color:#eedbb9}
.packet-tabs{display:flex;gap:6px;border-bottom:1px solid #444;margin:0 0 20px;position:sticky;top:0;background:#111;z-index:5;padding:10px 0}
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
.packet th{background:#222}.packet td,.packet th{border-color:#333}
.matchup{margin:8px 0 12px}
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
.matchup-head{display:grid;grid-template-columns:var(--gutter) minmax(0,1fr) var(--gutter);align-items:center;gap:var(--row-gap);margin-bottom:8px;font-size:15px;font-weight:600}
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
.stat-row{padding:3px 0;border-bottom:1px solid #ffffff09}
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
.packet .card{padding:12px 0}.packet .metrics{margin:12px 0;gap:24px}
.stat-bars{display:flex;gap:8px;height:5px;margin-top:12px}
.stat-bars span{border-radius:4px;min-width:0}
.stat-bars .left{background:var(--left-color,#888)}.stat-bars .right{background:var(--right-color,#888)}
.stat-bars span{box-shadow:0 0 0 1px #ffffff40}
.stat-impact{text-align:center;font-size:12px;color:#aaa;margin-top:8px}
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
.headline-table-light{background:#fff;color:#222;border-collapse:collapse;font-size:12px;font-family:Graduate,Georgia,serif;width:100%}
.headline-table-light th,.headline-table-light td{padding:2px 6px;border:1px solid #ddd;text-align:center;line-height:1.25;white-space:nowrap}
.headline-table-light th{background:#f2f2f2;font-weight:600}
.headline-table-light img{height:16px;width:16px;object-fit:contain;vertical-align:middle}
/* Column order is fixed in headline_table() -- numeric columns (spread/
   model/diff/sd, both markets) right-align like a real figures column
   instead of sitting dead-center; logo/date/team/pick columns stay
   centered. nth-child indices: 5,6=spread's line/model, 9,10=diff/sd,
   12,13=total's line/model, 14,15=total's diff/sd. */
.headline-table-light td:nth-child(5),.headline-table-light td:nth-child(6),.headline-table-light td:nth-child(9),
.headline-table-light td:nth-child(11),.headline-table-light td:nth-child(12),.headline-table-light td:nth-child(13){text-align:right}
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
.report-date{color:#aaa;margin:4px 0 18px;font-size:13px;font-family:Graduate,Georgia,serif}
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
.spec-card h2{margin-top:0}.spec-list{display:grid;grid-template-columns:minmax(120px,200px) minmax(0,1fr);gap:7px 18px;margin:0;font-size:13px;line-height:1.45}
.spec-list dt{color:#9ba8b5}.spec-list dd{margin:0;overflow-wrap:anywhere}.spec-list .spec-list{grid-template-columns:minmax(110px,170px) minmax(0,1fr)}
.spec-list ul{margin:0;padding-left:18px}
@media(max-width:650px){.spec-list,.spec-list .spec-list{grid-template-columns:1fr;gap:2px}.spec-list dd{margin-bottom:8px}}
.stats-table{background:#111;color:#e3e6e9;font:12px/1.25 Arial,sans-serif;font-variant-numeric:tabular-nums;border-collapse:collapse}
/* The headline sheet is all NFL font, headers and figures included. */
.headline-table{background:#111;color:#e3e6e9;font:13px/1.25 Graduate,Georgia,serif;font-variant-numeric:tabular-nums;border-collapse:collapse}
.headline-table{width:100%}
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
.packet .headline-table th{padding:7px 8px}
/* Confidence key, bottom right under the sheet (tier_legend). */
.tier-legend{display:flex;flex-direction:column;align-items:flex-end;gap:3px;margin:10px 0 0;font:11px Arial,sans-serif;color:#b7c0c9;text-align:right}
.tier-legend strong{font:12px Graduate,Georgia,serif;color:#e3e6e9}
.tier-key{display:flex;align-items:center;gap:6px}
.tier-key i{display:inline-block;width:10px;height:10px;border-radius:2px;flex:0 0 auto}
.tier-note{color:#8e99a5}
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
.tier-guide .tier-note{color:#8e99a5;margin:12px 0 0}
@media(max-width:650px){.tier-table th:nth-child(4),.tier-table td:nth-child(4),
.tier-table th:nth-child(5),.tier-table td:nth-child(5){display:none}.tier-guide summary{text-align:left}}
.packet .headline-table td{padding:3px 8px}
/* Every stats table formatted the same tight way (this used to be QB
   Elo-only, leaving the others visibly looser/wider) -- small logos,
   snug padding, all of them. */
.packet .stats-table th,.packet .stats-table td{padding:2px 6px}
.packet .headline-table th,.packet .stats-table th{background:#1a1d21;color:#aeb8c3;border:0;border-bottom:2px solid #41464e;white-space:nowrap}
.packet .headline-table td,.packet .stats-table td{border:0;border-bottom:1px solid #292d32;white-space:nowrap}
.headline-table tbody tr:hover,.stats-table tbody tr:hover{background:#1d2126}
.headline-table img{height:30px;width:34px;object-fit:contain;vertical-align:middle}
.stats-table img{height:20px;width:22px;object-fit:contain;vertical-align:middle}
.headline-table td:nth-child(4),.headline-table td:nth-child(7),
.headline-table-light td:nth-child(4),.headline-table-light td:nth-child(7){font-family:Graduate,Georgia,serif;font-size:14px}
.headline-table td:nth-child(5),.headline-table td:nth-child(6),.headline-table td:nth-child(9),
.headline-table td:nth-child(11),.headline-table td:nth-child(12),.headline-table td:nth-child(13){text-align:right}
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
.packet .pill.pick{background:#ffe590;color:#222}.pill.pick.tier-S{background:#e3c4ff}.pill.pick.tier-A{background:#b9e4c4}.sheet-notes{border-top:1px solid #333;margin-top:28px;padding-top:12px}.sheet-notes summary{cursor:pointer;font:15px Graduate,Georgia,serif}
.sheet-notes p{font-size:13px;line-height:1.5;color:#c8cdd3;margin:10px 0}.sheet-notes strong{color:#eee}
.pick-header{overflow-x:auto;border-bottom:1px solid #333;padding:10px 0 16px;margin-bottom:12px}
.pick-grid{display:grid;grid-template-columns:160px repeat(5,minmax(60px,1fr)) 160px;min-width:700px;align-items:center;gap:6px 8px;text-align:center;font-variant-numeric:tabular-nums}
.pick-label{font:10px Arial,sans-serif;color:#9ba8b5}.pick-label:first-child,.pick-label:nth-child(7){font:12px Graduate,Georgia,serif}.pick-label:first-child,.pick-qb{text-align:left}.pick-label:nth-child(7),.pick-qb.home{text-align:right}
.pick-team{display:flex;flex-wrap:wrap;align-items:center;gap:6px;font:24px Graduate,serif}.pick-team.home{justify-content:flex-end}.pick-team .logo{width:52px;height:52px}
.pick-value{font-size:15px;font-weight:600}.pick-qb{font:12px Graduate,Georgia,serif;white-space:nowrap}
.pick-score{grid-column:2/7;display:grid;grid-template-columns:1fr auto 1fr;gap:8px;align-items:baseline;font:12px Graduate,Georgia,serif;color:#9ba8b5}
.pick-score .score-label{justify-self:end}.pick-score .score-value{color:#eee}
summary.matchup-summary{padding-left:0}summary.matchup-summary:before{left:-12px}
.site-row{display:grid;grid-template-columns:var(--gutter) minmax(0,1fr) var(--gutter);gap:var(--row-gap);align-items:center;margin:12px 0;font-size:11px}
.site-label{white-space:nowrap}.site-label strong{margin-right:8px}.site-row .context-value{font-size:11px}
.site-row .context-value{text-align:right}
/* Weather: a heading, then one tight row per model input (matchup_attribution). */
.context-group{display:flex;justify-content:space-between;align-items:baseline;gap:8px;font-size:11px;font-weight:700;margin:14px 0 2px}
.context-group .context-value{font-size:11px;font-weight:400}
.site-row.weather-factor{margin:5px 0}.weather-factor .site-label{padding-left:12px;color:#ccc}
/* Closing line: baseline + every contribution + residual = the header's number. */
@media(max-width:650px){.packet{--gutter:130px;--val-w:55px;--row-gap:5px}.stat-line{grid-template-columns:calc(var(--gutter) - var(--val-w) - var(--row-gap)) var(--val-w) minmax(0,1fr) var(--gutter)}.matchup-head{grid-template-columns:var(--gutter) minmax(0,1fr) var(--gutter)}.site-label{white-space:normal}}
@media(max-width:650px){.packet-bundle{padding:10px}.packet-bundle .pkgtab-label{padding:8px;font-size:12px}.headline-table img{height:26px;width:28px}}
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
    starters = pd.concat([known[['season', 'week', f'{side}_team', f'{side}_qb_name']].rename(
        columns={f'{side}_team': 'team', f'{side}_qb_name': 'name'}) for side in ['away', 'home']])
    starters = starters.dropna(subset=['name']).sort_values(['season', 'week']).drop_duplicates('team', keep='last')
    starters['team'] = starters.team.replace(dc.RELOCATED_TEAMS)
    starters['name'] = starters.name.map(lambda n: utils.strip_suffix(f'{n.split()[0][0]}.{n.split()[1]}'))
    qb['name'] = qb.name.map(utils.strip_suffix)
    ratings = starters.merge(qb, on='name', how='left')[['team', 'weighted_qb_elo']].rename(
        columns={'weighted_qb_elo': 'off_qb_elo'})
    result = result.merge(ratings, on='team', how='left', validate='one_to_one').merge(
        defense, on='team', how='left', validate='one_to_one')
    utils.save_parquet(result, cached)
    return result


def stat_cell(stats, team, unit, metric, rank_before=False):
    column = f'{unit}_{metric}'
    if stats.empty or column not in stats or team not in stats.team.values:
        return '—'
    values = stats.set_index('team')[column].replace([np.inf, -np.inf], np.nan)
    value = values.loc[team]
    if pd.isna(value):
        return '—'
    formatted = f'{value:.1%}' if '%' in metric or metric.endswith('_pp') else f'{value:.1f}'
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
            label = 'QB Elo / allowed' if metric == 'qb_elo' else pretty(metric)
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
              'stuff_%': 'Runs stopped at / behind line', 'sack_%': 'Sacks / pass play',
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
    headings = ['Away', 'MarketLine' if market == 'spread' else 'Market O/U',
                'ModelLine' if market == 'spread' else 'Model O/U', 'Edge', 'SD', 'PICK', 'Home']
    labels = ''.join(f'<div class="pick-label">{h}</div>' for h in headings)
    if action == 'PASS':
        pick = '–'
    else:
        tier = pick_tier(market, row.week, row.get('total_game_importance'), float(np.sqrt(row.variance)),
                         row.get('market_base'))
        pick = f'<span class="pill pick tier-{tier}" title="{escape(tier)} confidence">{escape(action)}</span>'
    scores = ''
    if pd.notna(row.get('away_points')) and pd.notna(row.get('home_points')):
        # Three cells (label | score | spacer) so the score itself sits dead
        # center under the numbers, with its label just to the left.
        label = 'Implied score:' if row.get('scores_implied', False) == True else 'Prediction:'
        scores = (f'<span class="score-label">{label}</span>'
                  f'<span class="score-value">{away} {row.away_points:.1f} - {row.home_points:.1f} {home}</span><span></span>')
    return ('<header class="pick-header"><div class="pick-grid">' + labels
            + f'<div class="pick-team">{logo(row.away_team)}{away}</div>'
            + ''.join(f'<div class="pick-value">{v}</div>' for v in values)
            + f'<div class="pick-value">{pick}</div>'
            + f'<div class="pick-team home">{home}{logo(row.home_team)}</div>'
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


def specs_page(spec):
    """The Specs tab: model_spec's spec for this run, as labeled sections --
    the same content as the version folder's model.json, readable in the packet."""
    def label(key):
        return key.replace('_', ' ').capitalize()

    def render(value):
        if isinstance(value, dict):
            return '<dl class="spec-list">' + ''.join(
                f'<dt>{escape(label(k))}</dt><dd>{render(v)}</dd>' for k, v in value.items()) + '</dl>'
        if isinstance(value, list):
            short = all(len(str(v)) < 40 for v in value)
            return (escape(', '.join(map(str, value))) if short
                    else '<ul>' + ''.join(f'<li>{escape(str(v))}</li>' for v in value) + '</ul>')
        return escape(str(value))

    model = spec['model']
    head = (f'<h1>{escape(model["name"])} {escape(model["version"])}</h1>'
            f'<p class="report-date">Code {escape(model["code"])} · generated {escape(model["generated"])} · '
            'also saved as ../model.json for this model version</p>')
    sections = ''.join(f'<section class="card spec-card"><h2>{escape(label(key))}</h2>{render(value)}</section>'
                       for key, value in spec.items() if key != 'model')
    return head + sections


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
    'spread': dict(diff_cutoff=3.0, sd_cutoff=None),
    'total': dict(diff_cutoff=5.0, sd_cutoff=None),
}


# Confidence tiers. Every rule below was measured on ONE backtest -- Model 2.0
# with 'weighted' stats and a 20-week lookback, 2010-2025, 4363 games
# (TIER_SOURCE) -- and the same rules were checked against the carryover and
# steep-lookback-10 runs, which agree within a point or two. Break-even is
# 52.4%. A pick has to clear its market's edge first (PICK_TIERS['edge']);
# the tier then says what that rule has been worth historically.
#
# What drives the tiers is not the model's own confidence but the situation:
# how late in the season it is (in September the 20-week lookback is mostly
# last season), whether the ensemble agrees with itself (spread only), and
# where the market priced the total (totals only).
TIER_SOURCE = 'Model 2.0 (weighted, 20-week lookback) backtest, 2010-2025: 4363 games'
PICK_TIERS = {
    'spread': dict(edge=3.0, tiers=[
        ('S', lambda week, importance, sd, line: (13 <= week <= 14 or week >= 19) and (sd is None or sd <= 4.5)),
        ('B', lambda week, importance, sd, line: True)]),
    'total': dict(edge=5.0, tiers=[
        # 42-46 is a dead band for this model: those picks hit 47.8% while the
        # rest hit 59.8%, in all four eras and on every run. It shows up at
        # every edge level (48.1% even with no edge filter at all), and only on
        # the VEGAS total -- bucketing by the model's own number finds nothing.
        ('S', lambda week, importance, sd, line: 13 <= week <= 14 and not dead_total(line)),
        ('A', lambda week, importance, sd, line: week >= 5 and not dead_total(line)),
        ('B', lambda week, importance, sd, line: True)]),
}
DEAD_TOTAL = (42.0, 46.0)
TIER_COLORS = {'S': '#e3c4ff', 'A': '#b9e4c4', 'B': '#ffe590'}
# rule: what earns the tier. record/eras/volume: how it did in TIER_SOURCE.
TIER_GUIDE = {
    'spread': {
        'qualify': 'the model disagrees with the spread by at least 3 points',
        'S': dict(rule='week 13-14 or the playoffs, and the ensemble agrees with itself (SD at most 4.5)',
                  record='60.0% of 233 picks, +16.2% per bet', volume='about 15 a season',
                  eras='60 / 62 / 49 / 67% across four-season eras',
                  note='The SD condition earns its place: the same weeks without it are 58.0%, and the '
                       'high-SD games it drops are 53.9%. It holds in both halves of the record.'),
        'B': dict(rule='every other qualifying pick', record='49.1% of 2046 picks, -4.5% per bet',
                  volume='about 128 a season', eras='47 / 47 / 52 / 50%',
                  note='No era above break-even. Spreads before week 13 carry no measurable information: '
                       'the model disagreeing with the market there predicts nothing. Shown for reference.'),
    },
    'total': {
        'qualify': 'the model disagrees with the total by at least 5 points, and the posted total is outside 42-46',
        'S': dict(rule='week 13-14, posted total outside 42-46',
                  record='64.0% of 89 picks, +23.7% per bet', volume='about 6 a season',
                  eras='52 / 65 / 71 / 65%',
                  note='The smallest and strongest bucket; thin enough that a quiet season is normal.'),
        'A': dict(rule='week 5 on, posted total outside 42-46',
                  record='59.0% of 498 picks, +14.2% per bet', volume='about 31 a season',
                  eras='61 / 55 / 58 / 61%',
                  note='Above break-even in all four eras. A walk-forward check (rule picked on earlier '
                       'seasons only, applied to the next) chose the 42-46 skip in 12 of 12 seasons and '
                       'returned 59.9%, so this is not a hindsight fit.'),
        'B': dict(rule='weeks 1-4, or a posted total of 42-46 in any week',
                  record='49.5% of 570 picks, -3.8% per bet', volume='about 36 a season',
                  eras='47 / 53 / 52 / 47%',
                  note='Two different dead spots pooled: September, when the lookback is mostly last '
                       'season, and the 42-46 band, where this model is wrong in every era.'),
    },
}
TIER_RECORD = {market: {tier: guide[tier]['rule'] for tier in 'SAB' if tier in guide}
               for market, guide in TIER_GUIDE.items()}
TIER_HISTORY = {market: {tier: guide[tier]['record'].split(',')[0] for tier in 'SAB' if tier in guide}
                for market, guide in TIER_GUIDE.items()}


def dead_total(line):
    """The 42-46 band on the posted total, where this model has no edge."""
    return line is not None and pd.notna(line) and DEAD_TOTAL[0] <= float(line) <= DEAD_TOTAL[1]


def pick_tier(market, week, importance=None, sd=None, line=None):
    """S, A or B for a qualifying pick -- see PICK_TIERS and TIER_GUIDE.
    sd: the ensemble's disagreement. line: the market's own number (the posted
    total, or the spread). Either being None never blocks a tier."""
    for name, applies in PICK_TIERS[market]['tiers']:
        if applies(int(week), importance, sd, line):
            return name
    return 'B'


def tier_guide():
    """The expandable key under the sheet: every tier's rule and record."""
    sections = []
    for market, guide in TIER_GUIDE.items():
        rows = ''.join(
            f'<tr><td><span class="tier-chip" style="background:{TIER_COLORS[tier]}">{tier}</span></td>'
            f'<td>{escape(guide[tier]["rule"])}</td><td class="tier-record">{escape(guide[tier]["record"])}</td>'
            f'<td class="tier-record">{escape(guide[tier]["volume"])}</td>'
            f'<td class="tier-record">{escape(guide[tier]["eras"])}</td></tr>'
            f'<tr class="tier-note-row"><td></td><td colspan="4">{escape(guide[tier]["note"])}</td></tr>'
            for tier in 'SAB' if tier in guide)
        sections.append(
            f'<h4>{escape(market.title())} picks</h4>'
            f'<p class="tier-qualify">A pick appears when {escape(guide["qualify"])}.</p>'
            '<table class="tier-table"><thead><tr><th></th><th>Earns the tier</th><th>Record</th>'
            '<th>Volume</th><th>By era</th></tr></thead><tbody>' + rows + '</tbody></table>')
    return ('<details class="tier-guide"><summary>What the pick colours mean</summary>'
            + ''.join(sections)
            + f'<p class="tier-note">Measured on the {escape(TIER_SOURCE)}, and checked against the carryover '
              'and steep-lookback runs, which agree within a point or two. Break-even at -110 is 52.4%, so a '
              'tier below that is information, not an edge. Hit rates are what the rule did historically, not '
              'a forecast for any single pick.</p></details>')


def tier_legend(markets=('spread', 'total')):
    """The small colour key in the corner of the sheet."""
    items = ''.join(
        f'<span class="tier-key"><i style="background:{TIER_COLORS[tier]}"></i>{escape(tier)} '
        + escape(' · '.join(f'{market} {TIER_RECORD[market][tier]} {TIER_HISTORY[market][tier]}'
                            for market in markets if tier in TIER_RECORD[market]))
        + '</span>' for tier in ['S', 'A', 'B'])
    return (f'<div class="tier-legend"><strong>Confidence</strong>{items}'
            f'<span class="tier-note">Hit rates: {escape(TIER_SOURCE)}. Break-even is 52.4%.</span></div>')


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
    subtitle = f'{int(season)} Week {int(week)}'
    legend = [(TIER_COLORS[tier], f'{tier}  ' + ' · '.join(f'{market} {TIER_RECORD[market][tier]} {TIER_HISTORY[market][tier]}'
                                                            for market in ['spread', 'total'] if tier in TIER_RECORD[market]))
              for tier in ['S', 'A', 'B']]
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

    team_columns, right_columns = {3, 6}, {4, 5, 8, 10, 11, 12}  # 0-based; see headline_table's column order
    rows = []
    for row in (r for r in parsed.rows if r):
        styled = []
        for column, cell in enumerate(row):
            rule = styles.get(cell['id'], {})
            bold = cell['head'] or rule.get('font-weight') in ('600', '700', 'bold')
            team = not cell['head'] and column in team_columns
            size = 14 if team else 12
            styled.append(dict(
                cell, text=cell['text'].strip(), size=size,
                # The whole sheet is Graduate, which has one weight -- browsers
                # fake bold with a thicker stroke, so do the same here.
                font=font('graduate', size), stroke=bold,
                fill=rule.get('background-color', '#f2f2f2' if cell['head'] else '#ffffff'),
                color=rule.get('color', '#222222'),
                right=not cell['head'] and column in right_columns))
        rows.append(styled)

    pad_x, pad_y, grid, logo_px = 6 * scale, 2 * scale, scale, 16 * scale
    columns = max(len(row) for row in rows)
    widths = [0] * columns
    for row in rows:
        for column, cell in enumerate(row):
            content = logo_px if cell['logo'] else cell['font'].getlength(cell['text'])
            widths[column] = max(widths[column], int(content + 0.999) + 2 * pad_x)
    heights = [max(max(round(cell['size'] * 1.25 * scale), logo_px if cell['logo'] else 0) for cell in row) + 2 * pad_y
               for row in rows]

    margin = 16 * scale
    legend_font = font('sans', 9)
    title_font, subtitle_font = font('bold', 18), font('graduate', 13)
    title_h, subtitle_h = round(18 * 1.2 * scale), round(13 * 1.25 * scale)
    table_top = margin + title_h + 2 * scale + subtitle_h + 10 * scale
    table_w = sum(widths) + (columns + 1) * grid
    table_h = sum(heights) + (len(rows) + 1) * grid
    swatch, gap, line_gap = 8 * scale, 6 * scale, round(13 * scale)
    legend_w = max([legend_font.getlength(text) + swatch + gap for _, text in legend] or [0])
    legend_h = (len(legend) * line_gap + 8 * scale) if legend else 0
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
                anchor, left = ('rm', x + width - pad_x) if cell['right'] else ('mm', x + width / 2)
                draw.text((left, middle), cell['text'], font=cell['font'], fill=cell['color'], anchor=anchor,
                          stroke_width=scale // 2 if cell['stroke'] else 0, stroke_fill=cell['color'])
            x += width + grid
        y += height + grid
    # Confidence key under the table, right-aligned like the page's own.
    right = margin + max(table_w, legend_w)
    y = table_top + table_h + 8 * scale
    for color, text in legend:
        width = legend_font.getlength(text)
        box = right - width - swatch - gap
        draw.rectangle([box, y + 2 * scale, box + swatch, y + 2 * scale + swatch], fill=color, outline='#cccccc')
        draw.text((right, y), text, font=legend_font, fill='#444444', anchor='ra')
        y += line_gap
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


def headline_table(folder, light=False):
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
            frames[market] = settle(pd.read_csv(path), cutoffs['diff_cutoff'], cutoffs['sd_cutoff'])
    if not frames:
        return ''
    base = next(iter(frames.values()))
    sched = packet_schedule()[['season', 'week', 'away_team', 'home_team', 'gameday', 'gametime']]
    games = base[['season', 'week', 'away_team', 'home_team']].merge(
        sched, on=['season', 'week', 'away_team', 'home_team'], how='left')

    def market_fields(frame, g, market):
        match = frame[(frame.away_team == g.away_team) & (frame.home_team == g.home_team)] if frame is not None else None
        if match is None or match.empty:
            return dict(line=np.nan, model=np.nan, diff=np.nan, sd=np.nan, pick=''), None
        r = match.iloc[0]
        pick = (r.away_team if r.edge > 0 else r.home_team) if market == 'spread' else ('OVER' if r.edge > 0 else 'UNDER')
        pick = pick if bool(r.qualifies) else 'PASS'
        # market_base/prediction are stored away-minus-home; flip sign for
        # spread so both read as the away team's own line ("team -X" =
        # away favored by X), matching game_header's Market/Model rows.
        # Total has no team-perspective concept, so it's left alone.
        sign = -1 if market == 'spread' else 1
        return (dict(line=sign * r.market_base, model=sign * r.prediction, diff=abs(r.edge), sd=r.sd,
                     market_line=r.market_base, pick=pick), (pick if pick != 'PASS' else None))

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
        rows.append(dict(
            gameday=g.gameday, gametime=g.gametime,
            away_logo=g.away_team, away_team=g.away_team,
            spread=spread['line'], prediction=spread['model'],
            home_team=g.home_team, home_logo=g.home_team,
            diff=spread['diff'], pick=spread['pick'],
            total=total['line'], total_prediction=total['model'],
            total_diff=total['diff'], total_pick=total['pick'],
            spread_tier=pick_tier('spread', g.week, g.get('total_game_importance'), spread['sd'],
                                  spread['market_line']) if spread_pick else '',
            total_tier=pick_tier('total', g.week, g.get('total_game_importance'), total['sd'],
                                 total['market_line']) if total_pick else ''))
    table = pd.DataFrame(rows)
    table['gameday'] = pd.to_datetime(table.gameday)
    table['gametime'] = pd.to_datetime(table.gametime, format='%H:%M', errors='coerce').dt.time
    table = table.sort_values(['gameday', 'gametime', 'away_team']).reset_index(drop=True)
    # Tiers ride along for the sort, then step out of the displayed frame.
    tiers = table[['spread_tier', 'total_tier']]
    table = table.drop(columns=['spread_tier', 'total_tier'])

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
    if light:
        greens = LinearSegmentedColormap.from_list('headline_greens_light', ['#eafaf0', '#4caf7d'])
        reds = LinearSegmentedColormap.from_list('headline_reds_light', ['#fdf0ef', '#e2726b'])
    else:
        greens = LinearSegmentedColormap.from_list('headline_greens', ['#151b18', '#20513c'])
        reds = LinearSegmentedColormap.from_list('headline_reds', ['#1b1717', '#643536'])

    # Column order is fixed and documented here (also relied on by the
    # .headline-table nth-child CSS that right-aligns the numeric columns):
    # gameday/gametime/away_logo/away_team/spread/prediction/home_team/
    # home_logo/diff/pick, then O/U's total/total_prediction/total_diff/
    # total_pick. No SD column in either market -- the pick's tier colour
    # (PICK_TIERS) is the confidence signal now. relabel_index
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
        styled = styled.background_gradient(subset=gradient_diff, cmap=greens)
    if gradient_sd:
        styled = styled.background_gradient(subset=gradient_sd, cmap=reds)
    styled = (styled.format({
                 'gameday': lambda x: x.strftime('%a %m/%d'),
                 'gametime': lambda x: x.strftime('%I:%M %p').lstrip('0') if x else '—',
                 'spread': signed, 'prediction': signed, 'diff': plain,
                 'total': plain, 'total_prediction': plain, 'total_diff': plain,
                 'away_logo': lambda x: logo(x), 'home_logo': lambda x: logo(x),
                 'away_team': team_cell, 'home_team': team_cell,
             })
             .relabel_index(['Date', 'Time', '', 'Away', 'Spread', 'Model', 'Home', '', 'Diff', 'Picks',
                             'O/U', 'Model', 'Diff', 'Picks'], axis=1)
             .apply(tier_styles, axis=None))
    return styled.to_html() + tier_legend() + tier_guide()


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
        games = settle(games, cutoffs['diff_cutoff'], cutoffs['sd_cutoff'])
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
        travelling = 'attr_away_travel_adv' in games.columns
        notes = [
            # An SD condition is optional per market; neither uses one now.
            ('Picks', 'A pick needs an edge of at least '
                      + (f'{cutoffs["diff_cutoff"]:g} points'
                         + (f' and an SD of at most {cutoffs["sd_cutoff"]:.2f}' if cutoffs.get('sd_cutoff') else ''))
                      + '; otherwise PICK shows –. Its colour is the confidence tier: '
                      + ', '.join(f'{tier} = {TIER_RECORD[market][tier]}, {TIER_HISTORY[market][tier]} in the backtest'
                                  for tier in ['S', 'A', 'B'] if tier in TIER_RECORD[market])
                      + '. Break-even is 52.4%. Lines are stored market lines, not live odds, and starting QBs '
                        'aren’t verified.'),
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
                   + copy_picks_widget(season, week, headline_table(folder, light=True)) + headline_table(folder))
        (folder / 'index.html').write_text(page(title, packet_tabs('headline') + headline, 'packet headline-shell'), encoding='utf-8')
        (folder / 'stats.html').write_text(page(title, packet_tabs('stats') + '<h1>Stats</h1>'
            + stats_tables(snapshot, games, season, week, config['lookback'], weighted_snapshot), 'packet headline-shell'), encoding='utf-8')
        if config.get('spec'):
            (folder / 'specs.html').write_text(page(title, packet_tabs('specs') + specs_page(config['spec']), 'packet'),
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
    parser.add_argument('--lookback', type=int, default=20)
    parser.add_argument('--train-window', type=int, default=100, help='Two-sided only: training REG weeks')
    parser.add_argument('--model-version', choices=list(f'model_{v}' for v in model_spec.VERSIONS), default='model_2.0',
                        help='Which model version to build: model_2.0 (default), model_2.1 (adds the EPA inputs) '
                             'or model_2.2 (adds the travel-distance input). Each writes to its own '
                             'data/results/model_*/ folder.')
    parser.add_argument('--calculation', default='weighted',
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
        refresh_packet(args.season, args.week, args.lookback, shared=args.model != 'neural')
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
            neural_packet(args.season, args.week, args.lookback, args.iterations, args.seed)
