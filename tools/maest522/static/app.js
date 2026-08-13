const LABELS = [
  "Electronic---Minimal-Deep-Tech",
  "Electronic---Microhouse",
  "Electronic---RoMinimal",
];
const STATE_LABELS = {
  positive: "да",
  negative: "нет",
  uncertain: "не уверен",
  unreviewed: "не размечено",
};
const state = {
  projectId: Number(localStorage.getItem("maest522.projectId")) || null,
  activeLabel: localStorage.getItem("maest522.activeLabel") || LABELS[0],
  progress: [],
  goals: [],
  preflight: null,
  item: null,
  history: [],
};

const byId = (id) => document.getElementById(id);
const setupStatus = byId("setup-status");
const trustedStatus = byId("trusted-status");
const reviewStatus = byId("review-status");
const shortLabel = (label) => label.split("---", 2)[1];

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
  if (!state.projectId) throw new Error("Сначала создайте или откройте проект.");
  return state.projectId;
}

function showSetup(payload) {
  setupStatus.textContent = JSON.stringify(payload, null, 2);
}

function invalidatePreflight() {
  state.preflight = null;
  byId("trusted-commit").disabled = true;
  trustedStatus.textContent = "Параметры изменены. Выполните проверку заново.";
}

function renderProgress() {
  const container = byId("confirmed-progress");
  container.replaceChildren();
  state.progress.forEach((item) => {
    const card = document.createElement("article");
    card.className = `progress-card${item.complete ? " complete" : ""}`;
    const title = document.createElement("strong");
    title.textContent = shortLabel(item.label);
    const counts = document.createElement("span");
    counts.textContent = `Да ${item.positive_count}/${item.positive_target} · Нет ${item.negative_count}/${item.negative_target}`;
    const uncertain = document.createElement("small");
    uncertain.textContent = `Не уверен: ${item.uncertain_count}`;
    card.append(title, counts, uncertain);
    container.append(card);
  });

  const allComplete = state.progress.length === LABELS.length
    && state.progress.every((item) => item.complete);
  ["run-fingerprints", "freeze-splits", "export-manifest"].forEach((id) => {
    byId(id).disabled = !allComplete;
  });
  byId("finalization-note").textContent = allComplete
    ? "Все цели достигнуты. Можно построить fingerprints, один общий split и manifest."
    : "Сначала соберите цели «да» и «нет» для всех трёх жанров.";
}

function renderActiveGoal() {
  const goal = state.goals.find((item) => item.label === state.activeLabel);
  if (!goal) return;
  byId("positive-target").value = goal.positive_target;
  byId("negative-target").value = goal.negative_target;
}

function renderBatches(batches) {
  const container = byId("confirmed-batches");
  container.replaceChildren();
  if (!batches.length) {
    container.textContent = "Импортов пока нет.";
    return;
  }
  batches.forEach((batch) => {
    const row = document.createElement("article");
    row.className = "batch-row";
    const heading = document.createElement("strong");
    heading.textContent = `${STATE_LABELS[batch.state] || batch.state}: ${batch.new_count} новых`;
    const path = document.createElement("span");
    path.textContent = batch.source_path;
    const details = document.createElement("small");
    details.textContent = `найдено ${batch.discovered_count} · уже было ${batch.existing_count} · batch ${batch.batch_id}`;
    row.append(heading, path, details);
    container.append(row);
  });
}

async function refreshWorkspace() {
  if (!state.projectId) return;
  const projectId = requireProject();
  const encodedLabel = encodeURIComponent(state.activeLabel);
  const [progress, goals, batches] = await Promise.all([
    api(`/api/projects/${projectId}/confirmed-progress`),
    api(`/api/projects/${projectId}/goals`),
    api(`/api/projects/${projectId}/confirmed-batches/${encodedLabel}`),
  ]);
  state.progress = progress;
  state.goals = goals;
  renderProgress();
  renderActiveGoal();
  renderBatches(batches);
}

function renderReviewState() {
  document.querySelectorAll("#state-controls [data-state]").forEach((button) => {
    button.classList.toggle("selected", state.item?.state === button.dataset.state);
  });
}

function setReviewState(value) {
  if (!state.item) return;
  state.item.state = value;
  renderReviewState();
}

function renderItem() {
  byId("review-label").textContent = shortLabel(state.activeLabel);
  if (!state.item) {
    byId("track-name").textContent = "Нет выбранного трека";
    byId("track-context").textContent = "";
    byId("audio").removeAttribute("src");
    renderReviewState();
    return;
  }
  if (state.item.active_label !== state.activeLabel) {
    state.item = null;
    throw new Error("Сервер вернул очередь другого жанра; автоматическое переключение заблокировано.");
  }
  byId("track-name").textContent = state.item.filename;
  byId("queue-position").textContent = `Раунд ${state.item.round_number} · ${state.item.split}`;
  byId("track-context").textContent = state.item.source_labels.join(" · ");
  byId("notes").value = state.item.note || "";
  byId("audio").src = state.item.audio_url;
  renderReviewState();
}

async function loadNext() {
  try {
    const projectId = requireProject();
    const query = new URLSearchParams({ label: state.activeLabel });
    if (state.item) query.set("after_id", state.item.queue_item_id);
    const next = await api(`/api/projects/${projectId}/queue/next?${query}`);
    if (!next) {
      state.item = null;
      reviewStatus.textContent = `В очереди ${shortLabel(state.activeLabel)} нет незавершённых треков.`;
    } else {
      if (state.item) state.history.push(state.item.queue_item_id);
      state.item = next;
      reviewStatus.textContent = "";
    }
    renderItem();
  } catch (error) {
    reviewStatus.textContent = error.message;
  }
}

async function loadPrevious() {
  if (!state.history.length) return;
  try {
    const queueId = state.history.pop();
    const item = await api(`/api/projects/${requireProject()}/queue/${queueId}`);
    if (item.active_label !== state.activeLabel) {
      throw new Error("Предыдущий трек относится к другому активному жанру.");
    }
    state.item = item;
    renderItem();
  } catch (error) {
    reviewStatus.textContent = error.message;
  }
}

async function saveAndNext() {
  if (!state.item) return;
  if (state.item.state === "unreviewed") {
    reviewStatus.textContent = "Выберите «да», «нет» или «не уверен».";
    return;
  }
  try {
    await api(
      `/api/projects/${requireProject()}/queue/${state.item.queue_item_id}/annotations`,
      {
        method: "POST",
        body: JSON.stringify({
          label: state.activeLabel,
          state: state.item.state,
          note: byId("notes").value,
        }),
      },
    );
    await refreshWorkspace();
    await loadNext();
  } catch (error) {
    reviewStatus.textContent = error.message;
  }
}

function jumpAudio(ratio) {
  const audio = byId("audio");
  if (Number.isFinite(audio.duration)) audio.currentTime = audio.duration * ratio;
}

LABELS.forEach((label) => {
  const option = document.createElement("option");
  option.value = label;
  option.textContent = shortLabel(label);
  byId("active-label").append(option);
});
byId("active-label").value = state.activeLabel;
byId("project-id").textContent = state.projectId ? `Проект ${state.projectId}` : "Проект не выбран";
renderItem();

byId("active-label").addEventListener("change", async (event) => {
  state.activeLabel = event.target.value;
  localStorage.setItem("maest522.activeLabel", state.activeLabel);
  state.item = null;
  state.history = [];
  invalidatePreflight();
  renderItem();
  try { await refreshWorkspace(); }
  catch (error) { showSetup({ error: error.message }); }
});

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
    await refreshWorkspace();
  } catch (error) { showSetup({ error: error.message }); }
});

byId("save-goal").addEventListener("click", async () => {
  try {
    const result = await api(
      `/api/projects/${requireProject()}/goals/${encodeURIComponent(state.activeLabel)}`,
      {
        method: "PATCH",
        body: JSON.stringify({
          positive_target: Number(byId("positive-target").value),
          negative_target: Number(byId("negative-target").value),
        }),
      },
    );
    showSetup(result);
    await refreshWorkspace();
  } catch (error) { showSetup({ error: error.message }); }
});

["trusted-playlist-path", "trusted-playlist-state"].forEach((id) => {
  byId(id).addEventListener("change", invalidatePreflight);
});

byId("trusted-preflight").addEventListener("click", async () => {
  try {
    const result = await api(
      `/api/projects/${requireProject()}/trusted-playlists/preflight`,
      {
        method: "POST",
        body: JSON.stringify({
          playlist_path: byId("trusted-playlist-path").value,
          label: state.activeLabel,
          state: byId("trusted-playlist-state").value,
        }),
      },
    );
    state.preflight = result;
    byId("trusted-commit").disabled = !result.clean;
    trustedStatus.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    state.preflight = null;
    byId("trusted-commit").disabled = true;
    trustedStatus.textContent = error.message;
  }
});

byId("trusted-commit").addEventListener("click", async () => {
  try {
    if (!state.preflight?.clean) throw new Error("Сначала выполните чистый preflight.");
    const result = await api(
      `/api/projects/${requireProject()}/trusted-playlists/commit`,
      {
        method: "POST",
        body: JSON.stringify({
          playlist_path: state.preflight.playlist_path,
          label: state.preflight.label,
          state: state.preflight.state,
          expected_playlist_sha256: state.preflight.playlist_sha256,
        }),
      },
    );
    state.preflight = null;
    byId("trusted-commit").disabled = true;
    trustedStatus.textContent = JSON.stringify(result, null, 2);
    await refreshWorkspace();
  } catch (error) { trustedStatus.textContent = error.message; }
});

byId("import-source").addEventListener("click", async () => {
  try {
    const result = await api(`/api/projects/${requireProject()}/sources`, {
      method: "POST",
      body: JSON.stringify({
        source_path: byId("source-path").value,
        suggested_label: state.activeLabel,
        candidate_role: byId("candidate-role").value,
      }),
    });
    showSetup(result);
  } catch (error) { showSetup({ error: error.message }); }
});

byId("create-round").addEventListener("click", async () => {
  try {
    const result = await api(`/api/projects/${requireProject()}/rounds`, {
      method: "POST",
      body: JSON.stringify({
        label: state.activeLabel,
        round_number: Number(byId("round-number").value),
        split: byId("round-split").value,
      }),
    });
    showSetup(result);
    await loadNext();
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
    await refreshWorkspace();
  } catch (error) { showSetup({ error: error.message }); }
});
byId("export-manifest").addEventListener("click", async () => {
  try { showSetup(await api(`/api/projects/${requireProject()}/export`)); }
  catch (error) { showSetup({ error: error.message }); }
});

byId("load-next").addEventListener("click", loadNext);
byId("previous-item").addEventListener("click", loadPrevious);
byId("save-item").addEventListener("click", saveAndNext);
document.querySelectorAll("#state-controls [data-state]").forEach((button) => {
  button.addEventListener("click", () => setReviewState(button.dataset.state));
});
document.querySelectorAll("[data-jump]").forEach((button) => {
  button.addEventListener("click", () => jumpAudio(Number(button.dataset.jump)));
});

document.addEventListener("keydown", (event) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  const actions = {
    Space: () => byId("audio").paused ? byId("audio").play() : byId("audio").pause(),
    KeyP: () => setReviewState("positive"),
    KeyN: () => setReviewState("negative"),
    KeyU: () => setReviewState("uncertain"),
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

if (state.projectId) {
  refreshWorkspace().catch((error) => showSetup({ error: error.message }));
}
