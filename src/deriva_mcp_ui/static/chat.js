/* DERIVA Chatbot -- chat.js
 *
 * Handles:
 *   - session-info fetch to populate header and catalog mode
 *   - POST /chat with streaming SSE response via fetch + ReadableStream
 *   - Incremental Markdown rendering during stream
 *   - Persistent thinking indicator (loading dots below content until stream ends)
 *   - Tool call events rendered as collapsible <details> blocks
 *   - Paragraph break injected between text segments separated by tool calls
 *   - Loading indicator, error display, 401 -> logged-out state in page
 *   - Auto-resize textarea, Enter-to-send (Shift+Enter for newline)
 */

(function () {
  "use strict";

  const thread       = document.getElementById("thread");
  const input        = document.getElementById("message-input");
  const sendBtn      = document.getElementById("send-btn");
  const inputArea    = document.getElementById("input-area");
  const catalogBar   = document.getElementById("catalog-bar");
  const gpHostname   = document.getElementById("gp-hostname");
  const gpCatalogId  = document.getElementById("gp-catalog-id");
  const catalogTitle = document.getElementById("catalog-title");
  const userLabel    = document.getElementById("user-label");
  const logoutLink   = document.getElementById("logout-link");

  const LOADING_DOTS_HTML =
    '<span class="loading-dots"><span></span><span></span><span></span></span>';

  let busy = false;

  // ------------------------------------------------------------------
  // Logged-out state: show login link, hide input
  // ------------------------------------------------------------------

  function showLoggedOutState() {
    inputArea.style.display = "none";
    userLabel.textContent = "";
    logoutLink.textContent = "Log in";
    logoutLink.href = "login";

    const el = document.createElement("div");
    el.className = "msg msg-assistant";
    el.innerHTML = 'You are not logged in. <a href="login">Log in</a> to continue.';
    thread.appendChild(el);
  }

  // ------------------------------------------------------------------
  // Initialise: load session info
  // ------------------------------------------------------------------

  async function init() {
    let info;
    try {
      const resp = await fetch("session-info");
      if (resp.status === 401) { showLoggedOutState(); return; }
      if (!resp.ok) { appendError("Could not load session. Please refresh."); return; }
      info = await resp.json();
    } catch {
      appendError("Could not reach server. Please refresh.");
      return;
    }

    userLabel.textContent = info.display_name || info.user_id || "";

    if (info.catalog_mode === "default" && info.label) {
      catalogTitle.textContent = "DERIVA Chatbot: " + info.label;
    }

    if (info.catalog_mode === "general") {
      catalogBar.style.display = "flex";
    }
  }

  // ------------------------------------------------------------------
  // Send a message
  // ------------------------------------------------------------------

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || busy) return;

    busy = true;
    sendBtn.disabled = true;
    input.value = "";
    autoResize();

    appendUserMessage(text);
    const assistantEl = appendAssistantPlaceholder();

    const body = { message: text };
    if (catalogBar.style.display !== "none") {
      body.hostname   = gpHostname.value.trim();
      body.catalog_id = gpCatalogId.value.trim();
    }

    let accumulated = "";
    // Set when a tool_end arrives so the next text chunk gets a separator.
    // If the preceding text ended on a sentence boundary we use a paragraph
    // break; otherwise a plain space to avoid splitting mid-sentence words.
    let needsSeparator = false;

    try {
      const resp = await fetch("chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (resp.status === 401) { assistantEl.remove(); showLoggedOutState(); return; }
      if (!resp.ok) {
        assistantEl.remove();
        appendError(`Server error ${resp.status}`);
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamDone = false;

      while (!streamDone) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const sseEvents = [];
        let eol;
        while ((eol = buffer.indexOf("\n\n")) !== -1) {
          sseEvents.push(buffer.slice(0, eol));
          buffer = buffer.slice(eol + 2);
        }

        for (const sseEvent of sseEvents) {
          const lines = sseEvent.split("\n");
          let eventName = "message";
          let dataLine = "";
          for (const line of lines) {
            if (line.startsWith("event: ")) eventName = line.slice(7).trim();
            else if (line.startsWith("data: ")) dataLine = line.slice(6);
          }

          if (eventName === "done") { streamDone = true; break; }

          if (eventName === "error") {
            const payload = safeParseJSON(dataLine) || {};
            // Keep any partial content already rendered; only remove if nothing to show
            if (accumulated) {
              renderFinal(assistantEl, accumulated);
            } else if (!assistantEl.querySelector(".tool-call")) {
              assistantEl.remove();
            }
            if (payload.error === "auth") {
              showLoggedOutState();
            } else {
              appendError(payload.detail || "An error occurred.");
            }
            return;
          }

          if (eventName === "tool") {
            const toolData = safeParseJSON(dataLine);
            handleToolEvent(assistantEl, toolData);
            if (toolData && toolData.type === "tool_end") needsSeparator = true;
            continue;
          }

          // Default: text chunk
          const chunk = safeParseJSON(dataLine);
          if (typeof chunk === "string") {
            if (needsSeparator && accumulated) {
              // Paragraph break after a complete sentence; space after a mid-sentence split
              accumulated += /[.!?:]\s*$/.test(accumulated) ? "\n\n" : " ";
            }
            needsSeparator = false;
            accumulated += chunk;
            renderPartial(assistantEl, accumulated);
          }
        }
      }
    } catch (err) {
      if (accumulated) {
        renderFinal(assistantEl, accumulated);
      } else if (!assistantEl.querySelector(".tool-call")) {
        assistantEl.remove();
      }
      appendError("Connection lost: " + err.message);
      return;
    } finally {
      // Remove thinking indicator regardless of how the stream ended
      const thinkingEl = assistantEl.querySelector(".msg-thinking");
      if (thinkingEl) thinkingEl.remove();

      busy = false;
      sendBtn.disabled = false;
      input.focus();
    }

    // Final Markdown render
    if (accumulated) {
      renderFinal(assistantEl, accumulated);
    } else {
      // No text -- keep tool call blocks if present, otherwise remove placeholder
      const textEl = assistantEl.querySelector(".msg-text");
      if (textEl) textEl.remove();
      if (!assistantEl.querySelector(".tool-call")) {
        assistantEl.remove();
      }
    }
  }

  // ------------------------------------------------------------------
  // Tool call event rendering
  // ------------------------------------------------------------------

  function _updateToolsSummary(toolsEl) {
    const container = toolsEl.parentElement;
    const summary = container.querySelector(".msg-tools-container-summary");
    const total = toolsEl.querySelectorAll(".tool-call").length;
    const running = toolsEl.querySelectorAll(".tool-call-status").length -
                    [...toolsEl.querySelectorAll(".tool-call-status")]
                      .filter(s => s.textContent === "done").length;
    const label = `${total} tool call${total !== 1 ? "s" : ""}`;
    summary.textContent = running > 0 ? `${label} (${running} running)` : label;
  }

  function handleToolEvent(msgEl, data) {
    if (!data || !data.type) return;
    const toolsEl = msgEl.querySelector(".msg-tools");
    if (!toolsEl) return;
    const toolsContainer = toolsEl.parentElement;

    if (data.type === "tool_start") {
      // Reveal the container on first tool call
      toolsContainer.style.display = "";

      const callEl = document.createElement("details");
      callEl.className = "tool-call";
      callEl.dataset.toolName = data.name;

      const summaryEl = document.createElement("summary");
      summaryEl.className = "tool-call-summary";
      summaryEl.innerHTML =
        `<code>${escapeHtml(data.name)}</code>` +
        `<span class="tool-call-status">running</span>`;
      callEl.appendChild(summaryEl);

      const inputSection = document.createElement("div");
      inputSection.className = "tool-call-section";
      inputSection.innerHTML =
        `<div class="tool-call-label">Input</div>` +
        `<pre>${escapeHtml(JSON.stringify(data.input, null, 2))}</pre>`;
      callEl.appendChild(inputSection);

      toolsEl.appendChild(callEl);
      _updateToolsSummary(toolsEl);
      scrollToBottom();

    } else if (data.type === "tool_end") {
      // Find the most recent still-running call with this name
      const calls = [...toolsEl.querySelectorAll(`.tool-call[data-tool-name="${CSS.escape(data.name)}"]`)];
      const callEl = calls.reverse().find(
        el => el.querySelector(".tool-call-status")?.textContent === "running"
      );
      if (!callEl) return;

      callEl.querySelector(".tool-call-status").textContent = "done";

      const resultSection = document.createElement("div");
      resultSection.className = "tool-call-section";
      resultSection.innerHTML =
        `<div class="tool-call-label">Result</div>` +
        `<pre>${escapeHtml(data.result || "")}</pre>`;
      callEl.appendChild(resultSection);
      _updateToolsSummary(toolsEl);
      scrollToBottom();
    }
  }

  // ------------------------------------------------------------------
  // DOM helpers
  // ------------------------------------------------------------------

  function appendUserMessage(text) {
    const el = document.createElement("div");
    el.className = "msg msg-user";
    el.textContent = text;
    thread.appendChild(el);
    scrollToBottom();
  }

  function appendAssistantPlaceholder() {
    const el = document.createElement("div");
    el.className = "msg msg-assistant";

    // Collapsible container for all tool calls this turn (hidden until first tool fires)
    const toolsContainer = document.createElement("details");
    toolsContainer.className = "msg-tools-container";
    toolsContainer.open = false;
    toolsContainer.style.display = "none";

    const toolsContainerSummary = document.createElement("summary");
    toolsContainerSummary.className = "msg-tools-container-summary";
    toolsContainer.appendChild(toolsContainerSummary);

    const toolsEl = document.createElement("div");
    toolsEl.className = "msg-tools";
    toolsContainer.appendChild(toolsEl);
    el.appendChild(toolsContainer);

    // Text content (populated by renderPartial / renderFinal)
    const textEl = document.createElement("div");
    textEl.className = "msg-text";
    el.appendChild(textEl);

    // Persistent thinking indicator -- visible the entire time the stream is active,
    // even while tool calls are running or Claude is processing results.
    // Removed in the sendMessage finally block.
    const thinkingEl = document.createElement("div");
    thinkingEl.className = "msg-thinking";
    thinkingEl.innerHTML = LOADING_DOTS_HTML;
    el.appendChild(thinkingEl);

    thread.appendChild(el);
    scrollToBottom();
    return el;
  }

  function appendError(message) {
    const el = document.createElement("div");
    el.className = "msg msg-error";
    el.textContent = message;
    thread.appendChild(el);
    scrollToBottom();
  }

  function renderPartial(msgEl, text) {
    const textEl = msgEl.querySelector(".msg-text");
    if (!textEl) return;
    if (typeof marked !== "undefined") {
      textEl.innerHTML = marked.parse(text, { breaks: true, gfm: true });
    } else {
      textEl.textContent = text;
    }
    scrollToBottom();
  }

  function renderFinal(msgEl, text) {
    const textEl = msgEl.querySelector(".msg-text");
    if (!textEl) return;
    if (typeof marked !== "undefined") {
      textEl.innerHTML = marked.parse(text, { breaks: true, gfm: true });
    } else {
      textEl.textContent = text;
    }
    scrollToBottom();
  }

  function scrollToBottom() {
    thread.scrollTop = thread.scrollHeight;
  }

  // ------------------------------------------------------------------
  // Textarea auto-resize
  // ------------------------------------------------------------------

  function autoResize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 150) + "px";
  }

  // ------------------------------------------------------------------
  // Utility
  // ------------------------------------------------------------------

  function safeParseJSON(str) {
    try { return JSON.parse(str); } catch { return null; }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ------------------------------------------------------------------
  // Event listeners
  // ------------------------------------------------------------------

  sendBtn.addEventListener("click", sendMessage);

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener("input", autoResize);

  // ------------------------------------------------------------------
  // Boot
  // ------------------------------------------------------------------

  init();
})();
