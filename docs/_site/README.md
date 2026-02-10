# gemholm — Single-Page Paper/Project (Jekyll)  

📄 **A single-page academic/project showcase template**  
Includes Hero carousel, Abstract / Background / Approach / Results / Societal Impact sections, lightweight lightbox support, and *Patrick Hand + Chinese handwriting fonts*.  

---

## 🚀 Quick Start  

```bash
# Requires Ruby + Bundler
bundle install
bundle exec jekyll serve --livereload
```


## 🛠️ Edit Content

| Section         | File / Path                                                                |
| --------------- | -------------------------------------------------------------------------- |
| **Site Info**   | `_config.yml` → `title` / `description` / `links.*`                        |
| **Homepage**    | `index.html`                                                               |
| **Text Blocks** | `_includes/sections/{abstract,background,approach,results,impact}.md`      |
| **Hero Images** | `_data/hero.yml` (images in `assets/images/hero/`)                         |
| **Results**     | Configurable via `items` list                                              |
| **Styling**     | `assets/css/styles.scss` (e.g. `.section h2.hand-title` controls headings) |




## 🙏 Credits

Demo reference:
Ashish Vaswani et al., Attention Is All You Need, NeurIPS 2017.
We link to the original paper only; any reproduced tables/figures follow the copyright note in the PDF (scholarly/journalistic use).