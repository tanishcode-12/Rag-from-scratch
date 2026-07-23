(function () {
  "use strict";

  const form = document.getElementById("ask-form");
  const questionInput = document.getElementById("question");
  const submitBtn = document.getElementById("submit-btn");
  const formHint = document.getElementById("form-hint");
  const toggleButtons = document.querySelectorAll(".toggle-btn");

  const resultsSection = document.getElementById("results");
  const answerWithoutEl = document.getElementById("answer-without");
  const answerWithEl = document.getElementById("answer-with");
  const timeWithoutEl = document.getElementById("time-without");
  const timeWithEl = document.getElementById("time-with");
  const scopeListEl = document.getElementById("scope-list");
  const scopeEmbedderLabel = document.getElementById("scope-embedder-label");

  const errorBanner = document.getElementById("error-banner");
  const errorDetail = document.getElementById("error-detail");

  let selectedEmbedder = "bert";

  toggleButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      toggleButtons.forEach((b) => {
        b.classList.remove("is-active");
        b.setAttribute("aria-checked", "false");
      });
      btn.classList.add("is-active");
      btn.setAttribute("aria-checked", "true");
      selectedEmbedder = btn.dataset.embedder;
    });
  });

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.classList.toggle("is-loading", isLoading);
    formHint.textContent = isLoading
      ? "Loading models on first run can take a minute — subsequent questions are much faster."
      : "";
  }

  function showError(message) {
    errorDetail.textContent = message;
    errorBanner.hidden = false;
    resultsSection.hidden = true;
  }

  function clearError() {
    errorBanner.hidden = true;
  }

  function renderScope(chunks) {
    scopeListEl.innerHTML = "";

    if (!chunks.length) {
      const li = document.createElement("li");
      li.className = "scope-item";
      li.innerHTML = `<span></span><span class="scope-text">No passages retrieved.</span><span></span>`;
      scopeListEl.appendChild(li);
      return;
    }

    const distances = chunks.map((c) => c.distance);
    const maxDist = Math.max(...distances, 0.0001);

    chunks.forEach((chunk, i) => {
      // Lower L2 distance = more relevant. Invert so the bar fill visually
      // represents "closeness" rather than raw distance.
      const closeness = 1 - chunk.distance / maxDist;
      const li = document.createElement("li");
      li.className = "scope-item";
      li.innerHTML = `
        <span class="scope-rank">${i + 1}</span>
        <span class="scope-text">${escapeHtml(chunk.text)}</span>
        <span class="scope-bar-wrap">
          <span class="scope-bar-track">
            <span class="scope-bar-fill" style="width: ${Math.max(closeness, 0.05) * 100}%"></span>
          </span>
        </span>
      `;
      scopeListEl.appendChild(li);
    });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();

    const question = questionInput.value.trim();
    if (!question) return;

    setLoading(true);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, embedder: selectedEmbedder, top_k: 5 }),
      });

      const data = await response.json();

      if (!response.ok) {
        showError(data.error || "Unexpected server error.");
        return;
      }

      answerWithoutEl.textContent = data.answer_without_retrieval;
      answerWithEl.textContent = data.answer_with_retrieval;
      timeWithoutEl.textContent = `${data.timing_seconds.generation_without_retrieval}s`;
      timeWithEl.textContent = `${data.timing_seconds.generation_with_retrieval}s (+${data.timing_seconds.retrieval}s retrieval)`;
      scopeEmbedderLabel.textContent = `${data.embedder} · top ${data.retrieved_chunks.length}`;
      renderScope(data.retrieved_chunks);

      resultsSection.hidden = false;
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showError("Could not reach the server. Is the Flask app running?");
    } finally {
      setLoading(false);
    }
  });
})();
