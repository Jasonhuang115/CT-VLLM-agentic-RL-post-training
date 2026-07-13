const state = {
  sessionId: null,
  hasUploadedCt: false,
};

const els = {
  ctFiles: document.getElementById("ctFiles"),
  coordX: document.getElementById("coordX"),
  coordY: document.getElementById("coordY"),
  coordZ: document.getElementById("coordZ"),
  diameter: document.getElementById("diameter"),
  age: document.getElementById("age"),
  smoking: document.getElementById("smoking"),
  useTools: document.getElementById("useTools"),
  detectBtn: document.getElementById("detectBtn"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  status: document.getElementById("status"),
  messages: document.getElementById("messages"),
  chatForm: document.getElementById("chatForm"),
  message: document.getElementById("message"),
  ctLabel: document.querySelector(".field span"),
};

// ── 图片灯箱 ──────────────────────────────────────────
function initLightbox() {
  if (document.getElementById("lightbox")) return;
  const lb = document.createElement("div");
  lb.id = "lightbox";
  lb.className = "lightbox";
  lb.innerHTML = '<img /><span class="close">&times;</span>';
  lb.addEventListener("click", (e) => {
    if (e.target === lb || e.target.className === "close") lb.classList.remove("open");
  });
  document.body.appendChild(lb);
}

function openLightbox(src) {
  initLightbox();
  const lb = document.getElementById("lightbox");
  lb.querySelector("img").src = src;
  lb.classList.add("open");
}

// ── 工具函数 ──────────────────────────────────────────
function setStatus(text, isError = false) {
  els.status.textContent = text;
  els.status.classList.toggle("error", isError);
}

function appendMessage(role, content, extras = null) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  if (extras) bubble.appendChild(extras);
  article.appendChild(bubble);
  els.messages.appendChild(article);
  els.messages.scrollTop = els.messages.scrollHeight;
  return { article, bubble };
}

function appendStreamingBubble() {
  const article = document.createElement("article");
  article.className = "message assistant";
  const bubble = document.createElement("div");
  bubble.className = "bubble streaming";
  bubble.textContent = "";
  article.appendChild(bubble);
  els.messages.appendChild(article);
  els.messages.scrollTop = els.messages.scrollHeight;
  return { article, bubble };
}

function filesToForm(form) {
  const files = Array.from(els.ctFiles.files || []);
  // 多轮对话：已有缓存 CT，可以不传
  if (!files.length && state.hasUploadedCt) return;
  if (!files.length) throw new Error("请先上传 CT 文件。");
  files.forEach((file) => form.append("ct_files", file));
}

function coordPayload(required = true) {
  const x = Number(els.coordX.value);
  const y = Number(els.coordY.value);
  const z = Number(els.coordZ.value);
  if ([x, y, z].some((v) => Number.isNaN(v))) {
    if (required) throw new Error("请填写 X/Y/Z 世界坐标。");
    return null;
  }
  const coord = { x, y, z };
  const diameter = Number(els.diameter.value);
  if (!Number.isNaN(diameter) && els.diameter.value !== "") coord.diameter_mm = diameter;
  return coord;
}

function clinicalPayload() {
  const payload = {};
  const age = Number(els.age.value);
  if (!Number.isNaN(age) && els.age.value !== "") payload.age = age;
  if (els.smoking.value) payload.smoking_history = els.smoking.value;
  return payload;
}

function renderRoiImages(images) {
  const grid = document.createElement("div");
  grid.className = "roi-grid";
  for (const view of ["axial", "coronal", "sagittal"]) {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${images[view]}`;
    img.alt = view;
    img.addEventListener("click", () => openLightbox(img.src));
    const cap = document.createElement("figcaption");
    cap.textContent = view;
    fig.appendChild(img);
    fig.appendChild(cap);
    grid.appendChild(fig);
  }
  return grid;
}

function renderToolCalls(toolCalls) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `工具调用 ${toolCalls.length} 个`;
  details.appendChild(summary);
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(toolCalls, null, 2);
  details.appendChild(pre);
  return details;
}

// ── 流式分析 ──────────────────────────────────────────
async function streamAnalyze() {
  const message = els.message.value.trim() || "请分析这个肺结节。";
  const coord = coordPayload(!state.hasUploadedCt);

  // 用户消息提示
  let userLabel = message;
  if (coord) userLabel += `\n坐标: (${coord.x}, ${coord.y}, ${coord.z})`;
  appendMessage("user", userLabel);

  const { article, bubble } = appendStreamingBubble();

  setStatus("正在上传并生成 ROI...");

  const form = new FormData();
  try { filesToForm(form); } catch (e) { setStatus(e.message, true); return; }
  form.append("message", message);
  if (coord) form.append("nodule_coords", JSON.stringify([coord]));
  form.append("clinical_info", JSON.stringify(clinicalPayload()));
  form.append("use_tools", String(els.useTools.checked));
  if (state.sessionId) form.append("session_id", state.sessionId);

  const response = await fetch("/analyze/stream", { method: "POST", body: form });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(errText || `HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const dataStr = line.slice(6);
      let data;
      try { data = JSON.parse(dataStr); } catch { continue; }

      if (data.type === "status") {
        setStatus(data.text);
      } else if (data.type === "token") {
        bubble.textContent += data.text;
        bubble.classList.remove("streaming");
        void bubble.offsetWidth;
        bubble.classList.add("streaming");
        els.messages.scrollTop = els.messages.scrollHeight;
      } else if (data.type === "done") {
        state.sessionId = data.session_id;
        state.hasUploadedCt = true;
        bubble.classList.remove("streaming");

        const images = data.images || {};
        for (const [, imgs] of Object.entries(images)) {
          bubble.appendChild(renderRoiImages(imgs));
        }

        const toolCalls = data.tool_calls || {};
        const allCalls = Object.values(toolCalls).flat();
        if (allCalls.length) bubble.appendChild(renderToolCalls(allCalls));

        if (data.disclaimer) {
          const disc = document.createElement("details");
          disc.style.cssText = "margin-top:12px;border-top:1px solid var(--line);padding-top:10px";
          const s = document.createElement("summary");
          s.textContent = "免责声明";
          disc.appendChild(s);
          const pre = document.createElement("pre");
          pre.textContent = data.disclaimer;
          disc.appendChild(pre);
          bubble.appendChild(disc);
        }

        setStatus("分析完成");
        // 标记 CT 已上传，后续不再强制要求
        els.ctFiles.parentElement.classList.add("has-cache");
      }
    }
  }
}

// ── 非流式（兼容）─────────────────────────────────────
async function analyze() {
  const message = els.message.value.trim() || "请分析这个肺结节。";
  const coord = coordPayload(!state.hasUploadedCt);
  if (coord) appendMessage("user", `${message}\n坐标: (${coord.x}, ${coord.y}, ${coord.z})`);
  else appendMessage("user", message);
  setStatus("正在上传并生成 ROI...");

  const form = new FormData();
  try { filesToForm(form); } catch (e) { setStatus(e.message, true); return; }
  form.append("message", message);
  if (coord) form.append("nodule_coords", JSON.stringify([coord]));
  form.append("clinical_info", JSON.stringify(clinicalPayload()));
  form.append("use_tools", String(els.useTools.checked));
  if (state.sessionId) form.append("session_id", state.sessionId);

  const response = await fetch("/analyze", { method: "POST", body: form });
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { detail: text }; }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail);
  }
  state.sessionId = data.session_id;
  state.hasUploadedCt = true;
  els.ctFiles.parentElement.classList.add("has-cache");

  data.results.forEach((result) => {
    const extras = document.createElement("div");
    extras.appendChild(renderRoiImages(result.images));
    if (result.tool_calls?.length) extras.appendChild(renderToolCalls(result.tool_calls));
    appendMessage("assistant", result.report, extras);
  });
  setStatus("分析完成");
}

async function detect() {
  setStatus("正在调用自动检测器...");
  const form = new FormData();
  try { filesToForm(form); } catch (e) { setStatus(e.message, true); return; }
  if (state.sessionId) form.append("session_id", state.sessionId);
  const response = await fetch("/detect", { method: "POST", body: form });
  const text = await response.text();
  let detections;
  try { detections = JSON.parse(text); } catch { throw new Error(text); }
  if (!response.ok) throw new Error(JSON.stringify(detections.detail || detections));

  if (!detections.length) {
    setStatus("未检测到结节候选，请手动输入坐标。", true);
    return;
  }
  const first = detections[0];
  els.coordX.value = first.x;
  els.coordY.value = first.y;
  els.coordZ.value = first.z;
  if (first.diameter_mm) els.diameter.value = first.diameter_mm;
  setStatus(`已填入最高优先级候选，confidence=${first.confidence ?? "N/A"}`);
}

// ── 事件绑定 ──────────────────────────────────────────
els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try { await streamAnalyze(); } catch (error) {
    setStatus(error.message, true);
    appendMessage("assistant", `请求失败: ${error.message}`);
  }
});

els.analyzeBtn.addEventListener("click", async () => {
  try { await streamAnalyze(); } catch (error) {
    setStatus(error.message, true);
  }
});

els.detectBtn.addEventListener("click", async () => {
  try { await detect(); } catch (error) {
    setStatus(error.message, true);
  }
});
