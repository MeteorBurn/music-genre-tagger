const LABELS = [
  "Electronic---Microhouse",
  "Electronic---RoMinimal",
  "Electronic---DeepTech-Minimal",
];
const STATES = ["positive", "negative", "uncertain", "unreviewed"];
const STATE_LABELS = {
  positive: "да",
  negative: "нет",
  uncertain: "не уверен",
  unreviewed: "не размечено",
};
const state = {
  projectId: Number(localStorage.getItem("maest522.projectId")) || null,
  item: null,
  history: [],
  focusedLabel: 0,
};

const byId = (id) => document.getElementById(id);
const setupStatus = byId("setup-status");
const reviewStatus = byId("review-status");

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (response.status === 204) return null;
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

function requireProject() {
  if (!state.projectId) throw new Error("Create or open a project first.");
  return state.projectId;
}

function showSetup(payload) {
  setupStatus.textContent = JSON.stringify(payload, null, 2);
}

function renderLabels() {
  const grid = byId("label-grid");
  grid.replaceChildren();
  LABELS.forEach((label, labelIndex) => {
    const row = document.createElement("section");
    row.className = `label-row${labelIndex === state.focusedLabel ? " focused" : ""}`;
    const title = document.createElement("h3");
    title.textContent = `${labelIndex + 1}. ${label.split("---")[1]}`;
    row.append(title);
    const controls = document.createElement("div");
    controls.className = "state-controls";
    STATES.forEach((value) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.label = label;
      button.dataset.state = value;
      button.textContent = STATE_LABELS[value];
      button.classList.toggle("selected", state.item?.states[label] === value);
      button.addEventListener("click", () => setLabelState(labelIndex, value));
      controls.append(button);
    });
    row.append(controls);
    grid.append(row);
  });
}

function setLabelState(labelIndex, value) {
  if (!state.item) return;
  state.focusedLabel = labelIndex;
  state.item.states[LABELS[labelIndex]] = value;
  renderLabels();
}

function renderItem() {
  if (!state.item) {
    byId("track-name").textContent = "Нет выбранного трека";
    byId("audio").removeAttribute("src");
    renderLabels();
    return;
  }
  byId("track-name").textContent = state.item.filename;
  byId("queue-position").textContent = `Раунд ${state.item.round_number} · ${state.item.split}`;
  byId("track-context").textContent = state.item.source_labels.join(" · ");
  byId("notes").value = state.item.note || "";
  byId("audio").src = state.item.audio_url;
  renderLabels();
}

async function loadNext() {
  try {
    const projectId = requireProject();
    const after = state.item ? `?after_id=${state.item.queue_item_id}` : "";
    const next = await api(`/api/projects/${projectId}/queue/next${after}`);
    if (!next) {
      state.item = null;
      reviewStatus.textContent = "В очереди нет незавершённых треков.";
    } else {
      if (state.item) state.history.push(state.item.queue_item_id);
      state.item = next;
      state.focusedLabel = 0;
      reviewStatus.textContent = "";
    }
    renderItem();
    await refreshProgress();
  } catch (error) {
    reviewStatus.textContent = error.message;
  }
}

async function loadPrevious() {
  if (!state.history.length) return;
  try {
    const queueId = state.history.pop();
    state.item = await api(`/api/projects/${requireProject()}/queue/${queueId}`);
    renderItem();
  } catch (error) {
    reviewStatus.textContent = error.message;
  }
}

async function saveAndNext() {
  if (!state.item) return;
  if (Object.values(state.item.states).includes("unreviewed")) {
    reviewStatus.textContent = "Для всех трёх стилей выберите да, нет или не уверен.";
    return;
  }
  try {
    await api(
      `/api/projects/${requireProject()}/queue/${state.item.queue_item_id}/annotations`,
      {
        method: "POST",
        body: JSON.stringify({ states: state.item.states, note: byId("notes").value }),
      },
    );
    await loadNext();
  } catch (error) {
    reviewStatus.textContent = error.message;
  }
}

async function refreshProgress() {
  if (!state.projectId) return;
  const progress = await api(`/api/projects/${state.projectId}/progress`);
  byId("queue-position").textContent = `${progress.completed}/${progress.total} завершено`;
}

function jumpAudio(ratio) {
  const audio = byId("audio");
  if (Number.isFinite(audio.duration)) audio.currentTime = audio.duration * ratio;
}

LABELS.forEach((label) => {
  const option = document.createElement("option");
  option.value = label;
  option.textContent = label.split("---")[1];
  byId("source-label").append(option);
});
byId("project-id").textContent = state.projectId ? `Проект ${state.projectId}` : "Проект не выбран";
renderLabels();

byId("create-project").addEventListener("click", async () => {
  try {
    const result = await api("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name: byId("project-name").value }),
    });
    state.projectId = result.project_id;
    localStorage.setItem("maest522.projectId", String(state.projectId));
    byId("project-id").textContent = `Проект ${state.projectId}`;
    showSetup(result);
  } catch (error) { showSetup({ error: error.message }); }
});

byId("import-source").addEventListener("click", async () => {
  try {
    const result = await api(`/api/projects/${requireProject()}/sources`, {
      method: "POST",
      body: JSON.stringify({
        source_path: byId("source-path").value,
        suggested_label: byId("source-label").value,
        candidate_role: byId("candidate-role").value,
      }),
    });
    showSetup(result);
  } catch (error) { showSetup({ error: error.message }); }
});

byId("upload-playlist").addEventListener("click", async () => {
  try {
    const file = byId("playlist-file").files[0];
    if (!file) throw new Error("Выберите файл M3U/M3U8.");
    const result = await api(`/api/projects/${requireProject()}/sources`, {
      method: "POST",
      body: JSON.stringify({
        playlist_name: file.name,
        playlist_text: await file.text(),
        base_directory: byId("base-directory").value,
        suggested_label: byId("source-label").value,
        candidate_role: byId("candidate-role").value,
      }),
    });
    showSetup(result);
  } catch (error) { showSetup({ error: error.message }); }
});

byId("run-fingerprints").addEventListener("click", async () => {
  try { showSetup(await api(`/api/projects/${requireProject()}/fingerprints`, { method: "POST" })); }
  catch (error) { showSetup({ error: error.message }); }
});
byId("freeze-splits").addEventListener("click", async () => {
  try {
    showSetup(await api(`/api/projects/${requireProject()}/splits/freeze`, {
      method: "POST", body: JSON.stringify({ seed: 522 }),
    }));
  } catch (error) { showSetup({ error: error.message }); }
});
byId("create-round").addEventListener("click", async () => {
  try {
    showSetup(await api(`/api/projects/${requireProject()}/rounds`, {
      method: "POST",
      body: JSON.stringify({
        round_number: Number(byId("round-number").value),
        split: byId("round-split").value,
      }),
    }));
    await loadNext();
  } catch (error) { showSetup({ error: error.message }); }
});

byId("load-next").addEventListener("click", loadNext);
byId("previous-item").addEventListener("click", loadPrevious);
byId("save-item").addEventListener("click", saveAndNext);
document.querySelectorAll("[data-jump]").forEach((button) => {
  button.addEventListener("click", () => jumpAudio(Number(button.dataset.jump)));
});

document.addEventListener("keydown", (event) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  const actions = {
    Space: () => byId("audio").paused ? byId("audio").play() : byId("audio").pause(),
    Digit1: () => { state.focusedLabel = 0; renderLabels(); },
    Digit2: () => { state.focusedLabel = 1; renderLabels(); },
    Digit3: () => { state.focusedLabel = 2; renderLabels(); },
    KeyP: () => setLabelState(state.focusedLabel, "positive"),
    KeyN: () => setLabelState(state.focusedLabel, "negative"),
    KeyU: () => setLabelState(state.focusedLabel, "uncertain"),
    KeyX: () => setLabelState(state.focusedLabel, "unreviewed"),
    KeyJ: () => jumpAudio(0.2),
    KeyK: () => jumpAudio(0.5),
    KeyL: () => jumpAudio(0.8),
    Enter: saveAndNext,
    Backspace: loadPrevious,
  };
  if (actions[event.code]) {
    event.preventDefault();
    actions[event.code]();
  }
});
