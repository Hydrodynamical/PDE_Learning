"""
Expression sanitizer for the PDE benchmark.

Strips LaTeX wrappers, markdown fences, unicode junk, and other formatting
artifacts that LLMs inject around mathematical expressions.

INTEGRATION: Two places to hook this in.

═══════════════════════════════════════════════════════════════════════
OPTION 1: In main_loop.py — sanitize inside parse_probe_response()
═══════════════════════════════════════════════════════════════════════

Find the section in parse_probe_response() where QUERY specs are extracted.
It probably looks something like:

    if "QUERY:" in line.upper():
        spec = line.split(":", 1)[1].strip()
        actions.append({"action": "query", "spec": spec})

Change to:

    if "QUERY:" in line.upper():
        spec = line.split(":", 1)[1].strip()
        spec = sanitize_expression(spec)  # ← ADD THIS
        actions.append({"action": "query", "spec": spec})


═══════════════════════════════════════════════════════════════════════
OPTION 2: In the notebook — sanitize between parse and dispatch
═══════════════════════════════════════════════════════════════════════

In pde_single (cell 12), after line 31 and before line 41:

    actions = parse_probe_response(llm_text)           # line 31

    # Sanitize query expressions                       # ← ADD THIS BLOCK
    for a in actions:
        if a["action"] == "query" and "spec" in a:
            a["spec"] = sanitize_expression(a["spec"])

    response_text, done, prev_solve_coeffs, _ = dispatch_turn(...)  # line 41


═══════════════════════════════════════════════════════════════════════
ALSO FIX: Empty response guard (cell 12, around line 75)
═══════════════════════════════════════════════════════════════════════

Change:
    if llm_text is None:
        llm_text = "PREDICT:"

To:
    if not llm_text or not llm_text.strip():
        llm_text = "PREDICT:"

═══════════════════════════════════════════════════════════════════════
"""
import re


def sanitize_expression(expr: str) -> str:
    """Clean an LLM-generated math expression for the sympy/numpy parser.

    Handles:
      - LaTeX display/inline delimiters: \\(...\\), $...$, $$...$$
      - Markdown code fences and inline code: ```...```, `...`
      - Unicode junk: zero-width spaces, non-breaking spaces, smart quotes
      - Trailing punctuation, underscores, periods
      - Stray backslash commands: \\sin → sin, \\pi → pi, \\cdot → *, \\left/\\right
      - Curly-brace groups from LaTeX: x^{2} → x^2, \\frac{a}{b} → (a)/(b)
    """
    if not expr:
        return expr

    s = expr

    # ── 1. Strip outer wrappers ────────────────────────────────────────
    # Remove markdown code fences
    s = re.sub(r'^```\w*\s*', '', s)
    s = re.sub(r'\s*```$', '', s)

    # Remove inline backticks
    s = s.strip('`')

    # Remove LaTeX display math: $$ ... $$
    s = re.sub(r'^\$\$\s*', '', s)
    s = re.sub(r'\s*\$\$$', '', s)

    # Remove LaTeX inline math: $ ... $
    s = re.sub(r'^\$\s*', '', s)
    s = re.sub(r'\s*\$$', '', s)

    # Remove \( ... \) delimiters  (escaped for both \\( and raw \()
    s = re.sub(r'^\\+\(\s*', '', s)
    s = re.sub(r'\s*\\+\)$', '', s)

    # Remove \[ ... \] delimiters
    s = re.sub(r'^\\+\[\s*', '', s)
    s = re.sub(r'\s*\\+\]$', '', s)

    # ── 2. Truncate at first non-math character ─────────────────────────
    # GPT-5.4 sometimes appends training data garbage (Chinese gambling
    # spam, SEO text, etc.) after a valid expression on the same line.

    # 2a. Truncate at first non-printable-ASCII character
    m = re.search(r'[^\x20-\x7E]', s)
    if m:
        truncated = s[:m.start()].rstrip()
        if truncated:
            s = truncated

    # ── 3. Unicode cleanup ─────────────────────────────────────────────
    s = s.replace('\u200b', '')   # zero-width space
    s = s.replace('\u00a0', ' ')  # non-breaking space
    s = s.replace('\u2009', ' ')  # thin space
    s = s.replace('\u2013', '-')  # en-dash → minus
    s = s.replace('\u2014', '-')  # em-dash → minus
    s = s.replace('\u2212', '-')  # unicode minus → ASCII minus
    s = s.replace('\u00d7', '*')  # ×  → *
    s = s.replace('\u00b7', '*')  # ·  → *
    s = s.replace('\u2019', '')   # right single quote (stray apostrophe)
    s = s.replace('\u2018', '')   # left single quote

    # ── 3. LaTeX command cleanup ───────────────────────────────────────
    # Remove \left and \right (used around parens)
    s = re.sub(r'\\left\b', '', s)
    s = re.sub(r'\\right\b', '', s)

    # \cdot → *
    s = re.sub(r'\\cdot\b', '*', s)

    # \times → *
    s = re.sub(r'\\times\b', '*', s)

    # Common trig/math functions: \sin → sin, \cos → cos, \exp → exp, etc.
    for fn in ['sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'abs',
               'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']:
        s = re.sub(rf'\\{fn}\b', fn, s)

    # \pi → pi
    s = re.sub(r'\\pi\b', 'pi', s)

    # ── 4. LaTeX brace groups → plain parens ───────────────────────────
    # x^{2} → x^(2), x^{n+1} → x^(n+1)
    # Do this BEFORE stripping remaining backslashes
    s = re.sub(r'\^{([^}]*)}', r'^(\1)', s)

    # _{...} subscripts — usually not meaningful for expressions, strip them
    s = re.sub(r'_{([^}]*)}', r'', s)

    # \frac{a}{b} → (a)/(b)
    s = re.sub(r'\\frac{([^}]*)}{([^}]*)}', r'(\1)/(\2)', s)

    # Strip any remaining curly braces
    s = s.replace('{', '').replace('}', '')

    # ── 5. Strip remaining stray backslashes ───────────────────────────
    # After handling known commands, any remaining \X is probably junk
    s = re.sub(r'\\(?=[a-zA-Z])', '', s)  # \x where x is a letter
    s = s.replace('\\', '')  # lone backslashes

    # ── 5b. Truncate at first non-math word ───────────────────────────
    # Now that LaTeX is cleaned, detect appended garbage text.
    # Any 2+ letter word that isn't a known math identifier → truncate.
    _MATH_WORDS = {'x', 'pi', 'e',
                   'sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'abs',
                   'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh',
                   'asin', 'acos', 'atan'}
    for wm in re.finditer(r'(?<=[^a-zA-Z])([a-zA-Z]{2,})(?=[^a-zA-Z]|$)', s):
        if wm.group(1).lower() not in _MATH_WORDS:
            truncated = s[:wm.start()].rstrip(' \t*+,')
            if truncated:
                s = truncated
            break

    # ── 6. Implicit multiplication ─────────────────────────────────────
    # Insert * between: digit/letter and (, ) and letter/digit, ) and (
    # e.g. 2pi → 2*pi, x(1-x) → x*(1-x), )(  → )*(
    s = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', s)       # 2pi → 2*pi, 2( → 2*(
    s = re.sub(r'\)(\()', r')*(', s)                     # )( → )*(
    s = re.sub(r'\)([a-zA-Z0-9])', r')*\1', s)          # )x → )*x, )2 → )*2
    s = re.sub(r'([a-zA-Z0-9])\(', r'\1*(', s)          # x( → x*(  BUT preserve sin(, cos( etc
    # Spaces between identifiers → multiplication: pi x → pi*x
    s = re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z])', r'\1*\2', s)
    # Undo the insert for known functions: sin*( → sin(
    for fn in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'pi',
               'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'ln']:
        s = s.replace(f'{fn}*(', f'{fn}(')

    # ── 7. Trailing junk ───────────────────────────────────────────────
    s = s.rstrip('_.,;: \t\n\r')
    s = s.strip()

    # ── 8. Sanity: if we stripped everything, return original ──────────
    if not s:
        return expr.strip()

    return s


# ── Tests ──────────────────────────────────────────────────────────────

def _test():
    cases = [
        # (input, expected)
        (r'\(sin(pi*x)\)', 'sin(pi*x)'),
        (r'\\(sin(pi*x)\\)', 'sin(pi*x)'),
        ('$x^2*(1-x)$', 'x^2*(1-x)'),
        ('$$x*(1-x)*sin(pi*x)$$', 'x*(1-x)*sin(pi*x)'),
        ('`x*(1-x)`', 'x*(1-x)'),
        ('```python\nx*(1-x)\n```', 'x*(1-x)'),
        (r'x^{2}*(1-x)^{3}', 'x^(2)*(1-x)^(3)'),
        (r'\sin(\pi x)', 'sin(pi*x)'),
        (r'\sin(2\pi x)', 'sin(2*pi*x)'),
        (r'x \cdot (1-x)', 'x * (1-x)'),
        ('x*(1-x)_', 'x*(1-x)'),
        ('x*(1-x)\u200b', 'x*(1-x)'),
        (r'\frac{1}{2}*x*(1-x)', '(1)/(2)*x*(1-x)'),
        (r'\left(x\right)*\left(1-x\right)', '(x)*(1-x)'),
        # Already clean — should pass through unchanged
        ('sin(pi*x)', 'sin(pi*x)'),
        ('x**2*(1-x)', 'x**2*(1-x)'),
        ('x*(1-x)*(2*x-1)', 'x*(1-x)*(2*x-1)'),
    ]

    passed = 0
    for inp, expected in cases:
        result = sanitize_expression(inp)
        ok = result == expected
        passed += ok
        status = '✓' if ok else '✗'
        if not ok:
            print(f'  {status} sanitize({repr(inp)})')
            print(f'       got:      {repr(result)}')
            print(f'       expected: {repr(expected)}')
        else:
            print(f'  {status} {repr(inp):50s} → {repr(result)}')

    print(f'\n{passed}/{len(cases)} passed')


if __name__ == '__main__':
    _test()
