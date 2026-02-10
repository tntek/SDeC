<!-- background.md -->




This paper investigates the problem of ID shift in text-to-image (T2I) generation—where the same subject’s appearance changes across different generated scenes. While T2I models like Stable Diffusion excel at visual fidelity, they struggle to maintain consistent identity in narrative or sequential tasks. The authors attribute this to scene contextualization, a bias caused by the model’s learned association between subjects and typical scene contexts.
To address this, they introduce Scene De-Contextualization (SDeC), which mathematically reverses contextual bias through eigenvalue optimization using SVD. SDeC decouples identity embeddings from scene correlations, allowing consistent generation without needing full target-scene supervision. Experiments show that SDeC improves identity preservation and scene diversity across various models and tasks.





<div class="card pad tight">
  <p class="text-2">Visualization of how scene contextualization affects subject identity consistency in text-to-image generation.</p>

  <figure style="text-align:center;">
    <img src="{{ '/assets/images/background/bg-01.png' | relative_url }}" 
         alt="背景图" 
         style="max-width:100%; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,.08);">
<figcaption style="font-size:.95rem; color:#4b5563; margin-top:8px; text-align:left;">
  Figure 1: Illustration of scene contextualization with SDXL. <br>
  <b>Left:</b> The attire of the subject varies with the site. 
  <b>Right:</b> The subject’s clothing changes with the season.
</figcaption>

  </figure>
</div>
