(() => {
  const IDS = {
    button: "aio-summary-toggle",
    style: "aio-summary-toggle-style",
  };
  const MESSAGES = {
    toggle: "aio:toggle-summary",
    ready: "aio:summary-ready",
    disabled: "aio:summary-disabled",
    visible: "aio:summary-visible",
    hidden: "aio:summary-hidden",
  };

  let isReady = false;
  let isVisible = false;

  function ensureStyles() {
    if (document.getElementById(IDS.style)) return;

    const style = document.createElement("style");
    style.id = IDS.style;
    style.textContent = `
      #${IDS.button} {
        align-items: center;
        background: transparent;
        border: 1px solid hsl(var(--border, 214.3 31.8% 91.4%));
        border-radius: 6px;
        color: hsl(var(--foreground, 222.2 84% 4.9%));
        cursor: pointer;
        display: inline-flex;
        font: inherit;
        font-size: 13px;
        font-weight: 500;
        gap: 6px;
        height: 32px;
        margin-left: 8px;
        padding: 0 10px;
        white-space: nowrap;
      }

      #${IDS.button}:hover:not(:disabled),
      #${IDS.button}[aria-pressed="true"] {
        background: hsl(var(--accent, 210 40% 96.1%));
      }

      #${IDS.button}:disabled {
        cursor: not-allowed;
        opacity: 0.45;
      }

      #${IDS.button} .aio-summary-dot {
        border: 1.5px solid currentColor;
        border-radius: 999px;
        height: 8px;
        width: 8px;
      }

      #${IDS.button}[aria-pressed="true"] .aio-summary-dot {
        background: currentColor;
      }
    `;
    document.head.appendChild(style);
  }

  function findHeaderTarget() {
    const themeButton = Array.from(document.querySelectorAll("button")).find((button) => {
      const label = [
        button.getAttribute("aria-label"),
        button.getAttribute("title"),
        button.textContent,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      return label.includes("theme") || label.includes("light") || label.includes("dark");
    });

    if (themeButton?.parentElement) return themeButton.parentElement;
    return document.querySelector("header") || document.body;
  }

  function renderButton() {
    ensureStyles();

    const target = findHeaderTarget();
    let button = document.getElementById(IDS.button);
    if (!button) {
      button = document.createElement("button");
      button.id = IDS.button;
      button.type = "button";
      button.innerHTML = '<span class="aio-summary-dot"></span><span>Summary</span>';
      button.addEventListener("click", () => {
        if (!isReady) return;
        window.postMessage(MESSAGES.toggle, window.location.origin);
      });
    }

    button.disabled = !isReady;
    button.setAttribute("aria-label", isVisible ? "Hide summary" : "Show summary");
    button.setAttribute("aria-pressed", String(isVisible));
    button.title = isReady ? "Show or hide summary" : "Upload and process a file first";

    if (button.parentElement !== target) {
      target.appendChild(button);
    }
  }

  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;

    if (event.data === MESSAGES.ready) {
      isReady = true;
      isVisible = false;
      renderButton();
    }

    if (event.data === MESSAGES.disabled) {
      isReady = false;
      isVisible = false;
      renderButton();
    }

    if (event.data === MESSAGES.visible) {
      isReady = true;
      isVisible = true;
      renderButton();
    }

    if (event.data === MESSAGES.hidden) {
      isVisible = false;
      renderButton();
    }
  });

  const observer = new MutationObserver(renderButton);
  observer.observe(document.documentElement, { childList: true, subtree: true });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderButton);
  } else {
    renderButton();
  }
})();
