#!/usr/bin/env python3
# Conversor Markdown -> HTML (subconjunto usado em DOCUMENTACAO_ALGORITMO.md)
import sys, re, html

def slugify(text):
    t = re.sub(r'`([^`]*)`', r'\1', text)
    t = re.sub(r'\*\*([^*]*)\*\*', r'\1', t)
    t = t.strip().lower()
    out = []
    for ch in t:
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace() or ch == '-':
            out.append(' ')
    t = re.sub(r'\s+', '-', ''.join(out).strip())
    return re.sub(r'-+', '-', t)

def inline(text):
    codes = []
    def repl_code(m):
        codes.append(m.group(1)); return f'\x00C{len(codes)-1}\x00'
    text = re.sub(r'`([^`]+)`', repl_code, text)
    text = html.escape(text, quote=False)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\x00C(\d+)\x00', lambda m: '<code>' + html.escape(codes[int(m.group(1))]) + '</code>', text)
    return text

def render_list(items):
    res, stack = [], []
    for indent, ordered, text in items:
        tag = 'ol' if ordered else 'ul'
        if not stack or indent > stack[-1][0]:
            res.append('<' + tag + '>'); stack.append((indent, tag))
        elif indent < stack[-1][0]:
            while stack and stack[-1][0] > indent:
                res.append('</' + stack[-1][1] + '>'); stack.pop()
            if not stack:
                res.append('<' + tag + '>'); stack.append((indent, tag))
        res.append('<li>' + inline(text) + '</li>')
    while stack:
        res.append('</' + stack[-1][1] + '>'); stack.pop()
    return ''.join(res)

def convert(md):
    lines = md.split('\n')
    out, para, i = [], [], 0
    def flush():
        nonlocal para
        if para:
            out.append('<p>' + inline(' '.join(para)) + '</p>'); para = []
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            flush(); i += 1; code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i += 1
            i += 1
            out.append('<pre><code>' + html.escape('\n'.join(code)) + '</code></pre>'); continue
        s = line.strip()
        if s == '':
            flush(); i += 1; continue
        if re.fullmatch(r'-{3,}', s):
            flush(); out.append('<hr>'); i += 1; continue
        m = re.match(r'(#{1,6})\s+(.*)', line)
        if m:
            flush(); lv = len(m.group(1)); txt = m.group(2).strip()
            out.append('<h%d id="%s">%s</h%d>' % (lv, slugify(txt), inline(txt), lv)); i += 1; continue
        if '|' in line and i + 1 < len(lines) and '-' in lines[i+1] and re.match(r'^\s*\|?[\s:|-]+\|?\s*$', lines[i+1]):
            flush()
            hdr = [c.strip() for c in line.strip().strip('|').split('|')]
            i += 2; rows = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')]); i += 1
            t = ['<table><thead><tr>'] + ['<th>%s</th>' % inline(c) for c in hdr] + ['</tr></thead><tbody>']
            for r in rows:
                t.append('<tr>' + ''.join('<td>%s</td>' % inline(c) for c in r) + '</tr>')
            t.append('</tbody></table>'); out.append(''.join(t)); continue
        if s.startswith('>'):
            flush(); q = []
            while i < len(lines) and lines[i].strip().startswith('>'):
                q.append(lines[i].strip()[1:].strip()); i += 1
            out.append('<blockquote>' + inline(' '.join(q)) + '</blockquote>'); continue
        if re.match(r'^(\s*)([-*])\s+(.*)', line) or re.match(r'^(\s*)(\d+)\.\s+(.*)', line):
            flush(); items = []
            while i < len(lines):
                a = re.match(r'^(\s*)([-*])\s+(.*)', lines[i])
                b = re.match(r'^(\s*)(\d+)\.\s+(.*)', lines[i])
                if a:
                    items.append((len(a.group(1)), False, a.group(3)))
                elif b:
                    items.append((len(b.group(1)), True, b.group(3)))
                elif lines[i].strip() == '':
                    nxt = lines[i+1] if i + 1 < len(lines) else ''
                    if re.match(r'^(\s*)([-*])\s+', nxt) or re.match(r'^(\s*)\d+\.\s+', nxt):
                        i += 1; continue
                    break
                elif re.match(r'^\s{2,}\S', lines[i]) and items:
                    items[-1] = (items[-1][0], items[-1][1], items[-1][2] + ' ' + lines[i].strip()); i += 1; continue
                else:
                    break
                i += 1
            out.append(render_list(items)); continue
        para.append(s); i += 1
    flush()
    return '\n'.join(out)

CSS = """
@page { size: A4; margin: 17mm 15mm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; margin: 0; }
h1 { font-size: 20pt; margin: 0 0 .4em; }
h2 { font-size: 14.5pt; border-bottom: 1px solid #ddd; padding-bottom: 3px; margin: 1.5em 0 .6em; }
h3 { font-size: 12pt; margin: 1.2em 0 .4em; }
p { margin: .5em 0; }
code { font-family: Consolas, 'Courier New', monospace; background: #eef0f2; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #f6f8fa; border: 1px solid #e2e2e2; border-radius: 5px; padding: 9px 11px; font-size: 8.6pt; line-height: 1.4; white-space: pre-wrap; word-wrap: break-word; page-break-inside: avoid; margin: .7em 0; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: .8em 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #eef0f2; font-weight: 600; }
blockquote { border-left: 3px solid #b9c2cc; margin: .8em 0; padding: 4px 12px; color: #3d3d3d; background: #fafbfc; }
a { color: #0b5ed7; text-decoration: none; }
hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.5em 0; }
ul, ol { margin: .5em 0; padding-left: 1.6em; }
li { margin: .25em 0; }
h1, h2, h3 { page-break-after: avoid; }
"""

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding='utf-8') as f:
        md = f.read()
    body = convert(md)
    doc = '<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"><title>DDFacet</title><style>%s</style></head><body>%s</body></html>' % (CSS, body)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(doc)
    print('OK ->', dst)
