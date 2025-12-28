import os
from markdownify import MarkdownConverter

HTML_DIR = "html_dataset"
MD_DIR = "markdown_dataset"


class Clean(MarkdownConverter):
    def convert_script(self, el, text, parent_tags):
        return ''
    def convert_code(self, el, text, parent_tags):
        style = el.get('style', '') or ''
        if 'display' in style and 'none' in style:
            return ''
        return super().convert_code(el, text, parent_tags)


def convert_html_to_markdown(html_text: str) -> str:
    data = Clean(strip=['a','href','img','svg']).convert(html_text)
    return "\n".join(line.strip() for line in data.splitlines() if line.strip())


def process_all() -> None:
    os.makedirs(MD_DIR, exist_ok=True)
    for name in os.listdir(HTML_DIR):
        if not name.lower().endswith(('.html', '.htm')):
            continue
        html_path = os.path.join(HTML_DIR, name)
        if not os.path.isfile(html_path):
            continue
        base, _ = os.path.splitext(name)
        md_path = os.path.join(MD_DIR, f"{base}.md")
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()
        markdown = convert_html_to_markdown(html_content)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)


if __name__ == "__main__":
    process_all()
