#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strip_comments.py —— 递归删除文件夹（含子文件夹）中所有文本文件的注释

用法示例:
    python strip_comments.py                 # 处理脚本所在文件夹
    python strip_comments.py D:/mycode       # 处理指定文件夹
    python strip_comments.py -n              # 预览模式，只显示不修改
    python strip_comments.py -b              # 修改前生成 .bak 备份
    python strip_comments.py --only .py .js  # 只处理指定扩展名
"""

from __future__ import annotations

import argparse
import codecs
import os
import sys
from pathlib import Path

# =========================================================================
#  一、语言规则表
#     line    : 行注释起始标记
#     block   : 块注释 (开始, 结束)
#     strings : 字符串定界符（其中的内容不会被当作注释删除）
#     nested  : 块注释是否支持嵌套（如 Rust）
# =========================================================================

RULES: dict = {}


def _add(exts, line=(), block=(), strings=(), nested=False):
    """注册一批扩展名对应的注释规则"""
    rule = {
        'line': list(line),
        'block': list(block),
        'strings': list(strings),
        'nested': nested,
    }
    for e in exts:
        RULES[e.lower()] = dict(rule)


# --- C 系（C / C++ / ObjC / Java / C# / Kotlin / Swift / Scala / Dart ...）---
_add(
    ['.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hh', '.hxx', '.inl', '.ipp',
     '.m', '.mm', '.cu', '.cuh',
     '.java', '.cs', '.swift', '.kt', '.kts', '.scala', '.dart',
     '.groovy', '.gradle', '.proto', '.thrift',
     '.glsl', '.vert', '.frag', '.wgsl'],
    line=['//'],
    block=[('/*', '*/')],
    strings=['"', "'"],
)

# --- JavaScript / TypeScript ---
_add(
    ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'],
    line=['//'],
    block=[('/*', '*/')],
    strings=['"', "'", '`'],
)

# --- Go ---
_add(['.go'], line=['//'], block=[('/*', '*/')], strings=['"', '`'])

# --- Rust（块注释可嵌套）---
_add(['.rs'], line=['//'], block=[('/*', '*/')], strings=['"'], nested=True)

# --- PHP ---
_add(['.php', '.phtml', '.php3', '.php4', '.php5', '.phps'],
     line=['//', '#'], block=[('/*', '*/')], strings=['"', "'"])

# --- CSS / SCSS / LESS ---
_add(['.css'], block=[('/*', '*/')], strings=['"', "'"])
_add(['.scss', '.less'], line=['//'], block=[('/*', '*/')], strings=['"', "'"])

# --- HTML / XML ---
_add(['.html', '.htm', '.xml', '.xhtml', '.svg', '.xsd', '.xsl', '.xslt', '.rss'],
     block=[('<!--', '-->')])

# --- SQL ---
_add(['.sql', '.ddl', '.dml'],
     line=['--'], block=[('/*', '*/')], strings=['"', "'"])

# --- Shell ---
_add(['.sh', '.bash', '.zsh', '.ksh', '.fish'], line=['#'], strings=['"', "'"])

# --- Python ---
# 三引号（""" / '''）按“块注释”处理，等同于删除文档字符串；
# 普通单/双引号字符串仍然保留，不会被误删。
_add(
    ['.py', '.pyw'],
    line=['#'],
    block=[('"""', '"""'), ("'''", "'''")],
    strings=['"', "'"],
)

# --- Ruby / Perl ---
_add(['.rb', '.rake', '.gemspec'], line=['#'], strings=['"', "'"])
_add(['.pl', '.pm'], line=['#'], strings=['"', "'"])

# --- YAML / TOML ---
_add(['.yaml', '.yml'], line=['#'], strings=['"', "'"])
_add(['.toml'], line=['#'], block=[('"""', '"""'), ("'''", "'''")],
     strings=['"', "'"])

# --- INI / conf / properties ---
_add(['.ini', '.cfg', '.conf', '.properties'], line=['#', ';'], strings=['"', "'"])

# --- Lua ---
_add(['.lua'], line=['--'], block=[('--[[', ']]')], strings=['"', "'"])

# --- TeX ---
_add(['.tex', '.sty', '.cls'], line=['%'])

# --- R ---
_add(['.r'], line=['#'], strings=['"', "'"])

# --- PowerShell ---
_add(['.ps1', '.psm1', '.psd1'],
     line=['#'], block=[('<#', '#>')], strings=['"', "'"])

# --- Vim script ---
_add(['.vim'], line=['"'])

# --- 汇编 ---
_add(['.asm', '.s'], line=[';'])

# --- Pascal ---
_add(['.pas'], line=['//'], block=[('{', '}'), ('(*', '*)')], strings=["'"])

# --- Makefile ---
_add(['.mk', '.mak'], line=['#'], strings=['"', "'"])

# 无扩展名 / 特殊文件名
_FILENAME_RULE = {'line': ['#'], 'block': [], 'strings': ['"', "'"], 'nested': False}
FILENAME_RULES = {
    'makefile':    _FILENAME_RULE,
    'gnumakefile': _FILENAME_RULE,
    'dockerfile':  _FILENAME_RULE,
    'rakefile':    _FILENAME_RULE,
    'gemfile':     _FILENAME_RULE,
    '.gitignore':  _FILENAME_RULE,
    '.dockerignore': _FILENAME_RULE,
    '.env':        _FILENAME_RULE,
}


def get_rule(path: Path):
    """根据扩展名或文件名返回注释规则；返回 None 表示不处理"""
    rule = RULES.get(path.suffix.lower())
    if rule is None:
        rule = FILENAME_RULES.get(path.name.lower())
    return rule


# =========================================================================
#  二、核心：逐字符状态机删除注释
# =========================================================================

def strip_comments(text: str, rule: dict) -> str:
    """
    从 text 中删除 rule 描述的注释。
    会正确跳过字符串字面量，避免把 "http://x" 里的 // 当成注释。
    """
    line_marks   = rule.get('line') or []
    block_marks  = rule.get('block') or []
    string_marks = rule.get('strings') or []
    nested       = rule.get('nested', False)

    if not line_marks and not block_marks:
        return text

    # 所有标记按长度降序排列，保证 """ 优先于 "
    tokens = []
    for s in string_marks:
        tokens.append(('string', s, None))
    for s in line_marks:
        tokens.append(('line', s, None))
    for b in block_marks:
        tokens.append(('block', b[0], b[1]))
    tokens.sort(key=lambda t: -len(t[1]))

    first_chars = {t[1][0] for t in tokens}

    out: list = []
    i, n = 0, len(text)

    while i < n:
        ch = text[i]

        # 快速路径
        if ch not in first_chars:
            out.append(ch)
            i += 1
            continue

        matched = None
        for kind, tok, extra in tokens:
            if text.startswith(tok, i):
                matched = (kind, tok, extra)
                break

        if matched is None:
            out.append(ch)
            i += 1
            continue

        kind, tok, extra = matched

        # ---------- 字符串：原样保留 ----------
        if kind == 'string':
            out.append(tok)
            i += len(tok)
            while i < n:
                c = text[i]
                if c == '\\' and i + 1 < n:      # 转义字符
                    out.append(c)
                    out.append(text[i + 1])
                    i += 2
                    continue
                if text.startswith(tok, i):      # 字符串结束
                    out.append(tok)
                    i += len(tok)
                    break
                out.append(c)
                i += 1

        # ---------- 行注释：删到行尾 ----------
        elif kind == 'line':
            while out and out[-1] in ' \t':      # 去掉注释前的行尾空白
                out.pop()
            own_line = (not out) or out[-1] == '\n'
            j = text.find('\n', i)
            if j == -1:
                i = n
            else:
                if not own_line:                 # 行内有代码，保留换行
                    out.append('\n')
                i = j + 1

        # ---------- 块注释 ----------
        else:
            end = extra or tok
            j = i + len(tok)
            depth = 1
            while j < n:
                if nested and text.startswith(tok, j):
                    depth += 1
                    j += len(tok)
                elif text.startswith(end, j):
                    depth -= 1
                    j += len(end)
                    if depth == 0:
                        break
                else:
                    j += 1

            while out and out[-1] in ' \t':      # 去掉注释前的行尾空白
                out.pop()
            own_line = (not out) or out[-1] == '\n'
            # 注释独占一行时，连带它自己的换行一起吃掉，避免留下空行
            if own_line and j < n and text[j] == '\n':
                j += 1
            i = j

    return ''.join(out)


# =========================================================================
#  三、文件读写与遍历
# =========================================================================

DEFAULT_EXCLUDE_DIRS = {
    '.git', '.svn', '.hg', '.bzr',
    'node_modules', '__pycache__', '.mypy_cache', '.pytest_cache',
    'venv', '.venv', 'env', '.tox',
    'dist', 'build', 'target',
    '.idea', '.vscode', '.vs',
    'vendor', 'third_party', 'thirdparty',
}


def decode_bytes(data: bytes, preferred: str):
    """返回 (文本, 使用的编码, BOM字节)；无法解码返回 (None, None, None)"""
    bom = b''
    body = data
    if data.startswith(codecs.BOM_UTF8):
        bom = codecs.BOM_UTF8
        body = data[3:]

    for enc in (preferred, 'utf-8', 'gbk', 'latin-1'):
        try:
            return body.decode(enc), enc, bom
        except (UnicodeDecodeError, LookupError):
            continue
    return None, None, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='递归删除文件夹内所有文本文件的注释',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('root', nargs='?', default=None,
                        help='目标文件夹，默认为脚本所在文件夹')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='只预览，不修改任何文件')
    parser.add_argument('-b', '--backup', action='store_true',
                        help='修改前生成 .bak 备份')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示每个被修改的文件')
    parser.add_argument('--encoding', default='utf-8',
                        help='首选文件编码，默认 utf-8')
    parser.add_argument('--exclude-dir', nargs='+', default=[], metavar='DIR',
                        help='额外排除的目录名')
    parser.add_argument('--only', nargs='+', default=None, metavar='EXT',
                        help='只处理这些扩展名，例如 --only .py .js')
    parser.add_argument('--include-self', action='store_true',
                        help='允许处理脚本自身（默认跳过）')
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    root = Path(args.root).resolve() if args.root else script_path.parent

    if not root.is_dir():
        print(f'错误：{root} 不是一个有效的文件夹', file=sys.stderr)
        return 1

    exclude_dirs = DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir)
    only = {e.lower() for e in args.only} if args.only else None

    stats = {'scanned': 0, 'changed': 0, 'skipped': 0,
             'binary': 0, 'unknown': 0}

    for dirpath, dirnames, filenames in os.walk(root):
        # 原地过滤，os.walk 就不会进入这些目录
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs)

        for name in sorted(filenames):
            path = Path(dirpath) / name

            if path.resolve() == script_path and not args.include_self:
                continue

            ext = path.suffix.lower()
            if only is not None and ext not in only:
                continue

            rule = get_rule(path)
            if rule is None:
                stats['unknown'] += 1
                continue

            stats['scanned'] += 1

            try:
                raw = path.read_bytes()
            except OSError as e:
                print(f'[跳过] 无法读取 {path}: {e}', file=sys.stderr)
                stats['skipped'] += 1
                continue

            # 简单的二进制文件判断
            if b'\x00' in raw[:8192]:
                stats['binary'] += 1
                continue

            text, enc, bom = decode_bytes(raw, args.encoding)
            if text is None:
                print(f'[跳过] 编码无法识别 {path}', file=sys.stderr)
                stats['skipped'] += 1
                continue

            new_text = strip_comments(text, rule)
            if new_text == text:
                continue

            stats['changed'] += 1
            prefix = '[预览] ' if args.dry_run else '[修改] '
            if args.verbose or args.dry_run:
                print(prefix + str(path))

            if args.dry_run:
                continue

            if args.backup:
                try:
                    (path.parent / (path.name + '.bak')).write_bytes(raw)
                except OSError as e:
                    print(f'[警告] 备份失败 {path}: {e}', file=sys.stderr)

            try:
                path.write_bytes(bom + new_text.encode(enc))
            except OSError as e:
                print(f'[错误] 写入失败 {path}: {e}', file=sys.stderr)
                stats['changed'] -= 1

    print()
    print('=' * 56)
    print(f'根目录       : {root}')
    print(f'识别并处理   : {stats["scanned"]} 个文件')
    print(f'实际删除注释 : {stats["changed"]} 个文件'
          + ('（预览模式，未写入）' if args.dry_run else ''))
    if stats['binary']:
        print(f'跳过二进制   : {stats["binary"]} 个')
    if stats['unknown']:
        print(f'跳过未知类型 : {stats["unknown"]} 个')
    if stats['skipped']:
        print(f'跳过(读写失败): {stats["skipped"]} 个')
    print('=' * 56)

    return 0


if __name__ == '__main__':
    sys.exit(main())