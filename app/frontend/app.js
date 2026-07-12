const state = {
  sessionId: null,
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
};

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
  if (extras) {
    bubble.appendChild(extras);
  }
  article.appendChild(bubble);
  els.messages.appendChild(article);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function filesToForm(form) {
  const files = Array.from(els.ctFiles.files || []);
  if (!files.length) {
    throw new Error("请先上传 CT 文件。");
  }
  files.forEach((file) => form.append("ct_files", file));
}

function coordPayload(required = true) {
  const x = Number(els.coordX.value);
  const y = Number(els.coordY.value);
  const z = Number(els.coordZ.value);
  if ([x, y, z].some((v) => Number.isNaN(v))) {
    if (required) throw new Error("请填写 X/Y/Z 世界坐标，或先尝试自动检测。");
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

async function parseResponse(response) {
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { detail: text };
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail);
  }
  return data;
}

function renderRoiImages(images) {
  const grid = document.createElement("div");
  grid.className = "roi-grid";
  for (const view of ["axial", "coronal", "sagittal"]) {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${images[view]}`;
    img.alt = view;
    const cap = document.createElement("figcaption");
    cap.textContent = view;
    fig.appendChild(img);
    fig.appendChild(cap);
    grid.appendChild(fig);
  }
  return grid;
}

function renderResult(result, disclaimer) {
  const wrap = document.createElement("div");
  wrap.appendChild(renderRoiImages(result.images));

  const report = document.createElement("div");
  report.textContent = result.report;
  wrap.appendChild(report);

  if (result.tool_calls?.length) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `工具调用 ${result.tool_calls.length} 个`;
    details.appendChild(summary);
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(result.tool_calls, null, 2);
    details.appendChild(pre);
    wrap.appendChild(details);
  }

  const note = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "免责声明";
  note.appendChild(summary);
  const pre = document.createElement("pre");
  pre.textContent = disclaimer;
  note.appendChild(pre);
  wrap.appendChild(note);
  return wrap;
}

async function analyze() {
  const message = els.message.value.trim() || "请分析这个肺结节。";
  const coord = coordPayload(true);
  appendMessage("user", `${message}\n坐标: (${coord.x}, ${coord.y}, ${coord.z})`);
  setStatus("正在上传并生成 ROI...");

  const form = new FormData();
  filesToForm(form);
  form.append("message", message);
  form.append("nodule_coords", JSON.stringify([coord]));
  form.append("clinical_info", JSON.stringify(clinicalPayload()));
  form.append("use_tools", String(els.useTools.checked));
  if (state.sessionId) form.append("session_id", state.sessionId);

  const response = await fetch("/analyze", { method: "POST", body: form });
  const data = await parseResponse(response);
  state.sessionId = data.session_id;

  data.results.forEach((result) => {
    appendMessage("assistant", `结节 ${result.nodule_id}`, renderResult(result, data.disclaimer));
  });
  setStatus("分析完成");
}

async function detect() {
  setStatus("正在调用自动检测器...");
  const form = new FormData();
  filesToForm(form);
  const response = await fetch("/detect", { method: "POST", body: form });
  const detections = await parseResponse(response);
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

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await analyze();
  } catch (error) {
    setStatus(error.message, true);
    appendMessage("assistant", `请求失败: ${error.message}`);
  }
});

els.analyzeBtn.addEventListener("click", async () => {
  try {
    await analyze();
  } catch (error) {
    setStatus(error.message, true);
  }
});

els.detectBtn.addEventListener("click", async () => {
  try {
    await detect();
  } catch (error) {
    setStatus(error.message, true);
  }
});
