from markdownify import markdownify as md

# read txt file its a txt file with html content
with open("input_html_indeed.txt", "r", encoding="utf-8") as f:
    html_content = f.read()
data = md(html_content, strip=['a', 'href', 'img', 'svg'])
# clean up the markdown content, remove empty lines and leading/trailing whitespace
data = "\n".join(line.strip() for line in data.splitlines() if line.strip())
with open("input_markdown_indeed.txt", "w", encoding="utf-8") as f:
    f.write(data)
