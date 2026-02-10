<!-- ### Poster / Paper -->


<!-- Experimental results across a variety of benchmarks indicate that the proposed framework outperforms existing baselines in terms of predictive performance and computational efficiency. The evaluation also highlights the framework’s robustness under noisy conditions and its ability to generalize across tasks. -->

{% assign _pdf = site.data.results.pdf | default: site.links.pdf | default: '/assets/papers/paper.pdf' %}
{% assign pdf_url = _pdf %}
{% unless _pdf contains '://' %}{% assign pdf_url = _pdf | relative_url %}{% endunless %}
<div class="card pad">

<iframe
  class="pdf-embed"
  src="{{ pdf_url }}#toolbar=1&navpanes=0&view=FitH"
  loading="lazy">
</iframe>



  <!-- 兜底：如果浏览器禁用内联预览，显示下载/打开链接 -->
  <p style="margin-top:.75rem;text-align:center;">
    <a href="{{ pdf_url }}" target="_blank" rel="noopener">Click here to open/download the PDF</a>
  </p>
</div>


