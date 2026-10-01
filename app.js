/* EduGenie frontend - talks to the FastAPI backend with fetch(). */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);

  // ---------------------------------------------------------------- helpers
  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function inline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
    s = s.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    return s;
  }

  // Minimal, safe Markdown -> HTML (input is HTML-escaped first).
  function renderMarkdown(md) {
    const lines = String(md).replace(/\r\n/g, "\n").split("\n");
    let html = "";
    let list = null;
    let inCode = false;
    let code = [];
    let para = [];

    const flushPara = () => {
      if (para.length) {
        html += `<p>${inline(para.join(" "))}</p>`;
        para = [];
      }
    };
    const closeList = () => {
      if (list) {
        html += `</${list}>`;
        list = null;
      }
    };

    for (const raw of lines) {
      const line = raw.trimEnd();

      if (/^```/.test(line.trim())) {
        if (inCode) {
          html += `<pre><code>${esc(code.join("\n"))}</code></pre>`;
          code = [];
          inCode = false;
        } else {
          flushPara();
          closeList();
          inCode = true;
        }
        continue;
      }
      if (inCode) {
        code.push(raw);
        continue;
      }
      if (!line.trim()) {
        flushPara();
        closeList();
        continue;
      }

      let m;
      if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
        flushPara();
        closeList();
        const level = Math.min(m[1].length + 1, 5);
        html += `<h${level}>${inline(m[2])}</h${level}>`;
      } else if (/^\s*([-*_])\1{2,}\s*$/.test(line)) {
        flushPara();
        closeList();
        html += "<hr>";
      } else if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {
        flushPara();
        if (list !== "ul") {
          closeList();
          html += "<ul>";
          list = "ul";
        }
        html += `<li>${inline(m[1])}</li>`;
      } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
        flushPara();
        if (list !== "ol") {
          closeList();
          html += "<ol>";
          list = "ol";
        }
        html += `<li>${inline(m[1])}</li>`;
      } else {
        closeList();
        para.push(line.trim());
      }
    }
    if (inCode) html += `<pre><code>${esc(code.join("\n"))}</code></pre>`;
    flushPara();
    closeList();
    return html;
  }

  async function api(url, options) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (e) {
      throw new Error("Cannot reach the EduGenie server. Is it still running?");
    }
    let data = null;
    try {
      data = await res.json();
    } catch (e) {
      /* non-JSON response */
    }
    if (!res.ok) {
      throw new Error((data && data.error) || `Request failed (${res.status}).`);
    }
    return data;
  }

  const postJSON = (url, body) =>
    api(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

  function showLoading(out, text) {
    out.innerHTML = `<div class="loading"><div class="spinner"></div><span>${esc(text)}</span></div>`;
  }

  function showError(out, message) {
    out.innerHTML = "";
    const div = document.createElement("div");
    div.className = "error";
    div.textContent = "⚠️ " + message;
    out.appendChild(div);
  }

  function showResult(out, markdown, tag) {
    out.innerHTML = `<div class="result">${renderMarkdown(markdown)}</div>`;
    if (tag) {
      const t = document.createElement("span");
      t.className = "tag";
      t.textContent = tag;
      out.firstChild.appendChild(t);
    }
  }

  // Wire a form: validate -> loading -> call -> render.
  function wire(formId, outId, loadingText, handler) {
    const form = $(formId);
    const out = $(outId);
    const btn = $("button[type='submit']", form);
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      btn.disabled = true;
      showLoading(out, loadingText);
      try {
        await handler(out);
      } catch (e) {
        showError(out, e.message);
      } finally {
        btn.disabled = false;
      }
    });
    // Ctrl/Cmd+Enter submits from textareas
    form.querySelectorAll("textarea").forEach((ta) =>
      ta.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") form.requestSubmit();
      })
    );
  }

  // ------------------------------------------------------------------- tabs
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $("#panel-" + tab.dataset.tab).classList.add("active");
    });
  });

  // ----------------------------------------------------------------- status
  (async () => {
    const el = $("#status");
    try {
      const h = await api("/health");
      if (h.gemini_configured) {
        el.textContent = "Gemini connected";
        el.className = "status ok";
      } else {
        el.textContent = "Set GEMINI_API_KEY in .env";
        el.className = "status warn";
      }
    } catch (e) {
      el.textContent = "server unreachable";
      el.className = "status warn";
    }
  })();

  // ------------------------------------------------------------------- Q&A
  wire("#qaForm", "#qaResult", "Thinking…", async (out) => {
    const q = $("#question").value.trim();
    const data = await api("/qa?question=" + encodeURIComponent(q));
    showResult(out, data.answer);
  });

  // ----------------------------------------------------------- Explanation
  wire("#explainForm", "#explanationResult", "Preparing an explanation (the first run loads the local model and can take a minute)…", async (out) => {
    const topic = $("#topic").value.trim();
    const data = await postJSON("/explain", { topic });
    showResult(out, data.explanation, data.source ? `Generated by: ${data.source === "local" ? "LaMini-Flan-T5 (local)" : "Gemini"}` : "");
  });

  // --------------------------------------------------------------- Summary
  wire("#summaryForm", "#summaryResult", "Summarizing…", async (out) => {
    const text = $("#summaryText").value.trim();
    const data = await postJSON("/summarize", { text });
    showResult(out, data.summary);
  });

  // ------------------------------------------------------------------ Quiz
  function renderQuiz(out, quiz) {
    out.innerHTML = "";
    let answered = 0;
    let score = 0;

    quiz.forEach((item, idx) => {
      const card = document.createElement("div");
      card.className = "q-card";

      const h = document.createElement("h4");
      h.textContent = `Q${idx + 1}. ${item.question}`;
      card.appendChild(h);

      const feedback = document.createElement("div");
      feedback.className = "feedback";

      item.options.forEach((option) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "opt";
        b.textContent = option;
        b.addEventListener("click", () => {
          const buttons = card.querySelectorAll(".opt");
          buttons.forEach((x) => (x.disabled = true));
          if (option === item.answer) {
            b.classList.add("correct");
            feedback.classList.add("good");
            feedback.textContent = "✅ Correct!";
            score += 1;
          } else {
            b.classList.add("wrong");
            buttons.forEach((x) => {
              if (x.textContent === item.answer) x.classList.add("correct");
            });
            feedback.classList.add("bad");
            feedback.textContent = `❌ Not quite. The correct answer is: ${item.answer}`;
          }
          answered += 1;
          if (answered === quiz.length) {
            const s = document.createElement("div");
            s.className = "score";
            s.textContent = `🎉 You scored ${score} / ${quiz.length}`;
            out.appendChild(s);
          }
        });
        card.appendChild(b);
      });

      card.appendChild(feedback);
      out.appendChild(card);
    });
  }

  wire("#quizForm", "#quizResult", "Generating quiz…", async (out) => {
    const text = $("#quizText").value.trim();
    const data = await postJSON("/quiz", { text });
    renderQuiz(out, data.quiz);
  });

  // ---------------------------------------------------------- Learning path
  wire("#learnForm", "#learnResult", "Building your learning path…", async (out) => {
    const topic = $("#learnTopic").value.trim();
    const data = await api("/learn/recommendations?topic=" + encodeURIComponent(topic));
    showResult(out, data.recommendation);
  });
})();
