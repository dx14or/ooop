const runBtn = document.getElementById("runBtn");
const dataPathInput = document.getElementById("dataPath");
const results = document.getElementById("results");

function setResults(items, warning) {
  results.innerHTML = "";
  if (warning) {
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = `Warning: ${warning}`;
    results.appendChild(note);
  }
  if (!items.length) {
    results.innerHTML = "<p class=\"hint\">No predictions yet.</p>";
    return;
  }

  items.forEach((item, index) => {
    const card = document.createElement("div");
    card.className = "result-card";
    const label = item.label || item.terms || `Topic ${item.topic_id}`;
    card.innerHTML = `
      <p class="result-title">#${index + 1} ${label}</p>
      <p class="result-meta">Probability: ${(item.prob * 100).toFixed(1)}%</p>
      ${item.terms ? `<p class="result-meta">Top terms: ${item.terms}</p>` : ""}
    `;
    results.appendChild(card);
  });
}

runBtn.addEventListener("click", async () => {
  const dataPath = dataPathInput.value.trim();
  results.innerHTML = "<p class=\"hint\">Running prediction...</p>";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_path: dataPath }),
    });

    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.detail || "Request failed");
    }

    const data = await response.json();
    setResults(data.predictions || [], data.warning);
  } catch (error) {
    results.innerHTML = `<p class="hint">Error: ${error.message}</p>`;
  }
});
