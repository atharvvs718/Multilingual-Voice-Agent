// What the page does: shows the notes and conversations the server holds, sends questions that are
// typed or spoken, and plays the replies. The server is server.py; every call goes to /api.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const el = (tag, className = "", text = null) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== null) node.textContent = text;
    return node;
  };
  const icon = (shape, className = "") => {
    const holder = document.createElement("template");
    holder.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true" class="${className}">${shape}</svg>`;
    return holder.content.firstChild;
  };
  const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const clock = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
  const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;

  const ICONS = {
    file: '<path d="M7 3h7l4 4v14H7zM14 3v4h4M10 12h5M10 16h5"/>',
    mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    arrow: '<path d="M12 19V5m0 0-6 6m6-6 6 6"/>',
    play: '<path d="M8 5.5v13l11-6.5z"/>',
    pause: '<path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/>',
    copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h8"/>',
    up: '<path d="M7 11v9H4v-9zM7 11l4-7c1.5 0 2.5 1 2.5 2.5V10H18a2 2 0 0 1 2 2.3l-1 6A2 2 0 0 1 17 20H7"/>',
    down: '<path d="M7 13V4H4v9zM7 13l4 7c1.5 0 2.5-1 2.5-2.5V14H18a2 2 0 0 0 2-2.3l-1-6A2 2 0 0 0 17 4H7"/>',
  };
  const MATH = [
    { left: "$$", right: "$$", display: true },
    { left: "\\[", right: "\\]", display: true },
    { left: "$", right: "$", display: false },
    { left: "\\(", right: "\\)", display: false },
  ];
  const SPEEDS = [1, 1.25, 1.5];
  const SPRING = "cubic-bezier(0.34, 1.4, 0.64, 1)";
  const NO_NOTES = $("summary").textContent;
  const DROP_HINT = $("drop-hint").textContent;
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const app = $("app");
  const leaf = $("leaf");
  const talk = $("talk");
  const composer = $("composer");
  const field = $("ask");
  const send = $("send");
  const mic = $("mic");

  const state = {
    language: ["mr", "hi", "en"].includes(localStorage.getItem("language")) ? localStorage.getItem("language") : "mr",
    voice: localStorage.getItem("voice") || "",
    voices: [],
    notebook: localStorage.getItem("notebook") || "",       // the open notebook; the server picks the first if this one is gone
    notebooks: [],      // {id, title}, oldest first
    chat: "",           // the open conversation; "" until its first question is answered
    chats: [],          // {id, title}, newest first
    asked: 0,           // counts questions, so the voice of an overtaken reply is not played
    waiting: false,     // a question is being answered
    hasNotes: false,
    maxClipMs: 30000,
    maxQuestionChars: 1000,
    maxUploadMb: 25,
  };

  // ---------- Talking to the server ----------

  async function api(path, options) {
    const response = await fetch(path, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = body.error || body.message || (typeof body.detail === "string" ? body.detail : "") || "Something went wrong. Please try again.";
      throw Object.assign(new Error(message), { body });
    }
    return body;
  }

  const sendJson = (path, body) =>
    api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  // The lines of JSON an /api/ask answer is made of, as they arrive.
  async function* events(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let end;
      while ((end = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, end).trim();
        buffer = buffer.slice(end + 1);
        if (line) yield JSON.parse(line);
      }
    }
  }

  const stateOf = (notebook) => api(`/api/state?language=${state.language}&notebook=${encodeURIComponent(notebook)}`);

  let toastTimer = 0;
  function toast(message) {
    const box = $("toast");
    box.textContent = message;
    box.hidden = true;
    void box.offsetWidth;       // restarts the entrance animation
    box.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (box.hidden = true), 3800);
  }

  // ---------- Notes: the bookshelf and the chapter heading ----------

  const chaptersLine = (chapters) =>
    !chapters.length ? "" : chapters.length === 1 ? `Chapter ${chapters[0]}` : `Chapters ${chapters[0]}–${chapters[chapters.length - 1]}`;

  function showNotes(notes) {
    const sources = notes.sources || [];
    state.hasNotes = sources.length > 0;
    $("sources").replaceChildren(
      ...sources.map((source) => {
        const row = el("li");
        const name = el("b", "", source.source);
        name.title = source.source;
        row.append(icon(ICONS.file), name, el("i", "", [plural(source.pages, "page"), chaptersLine(source.chapters)].filter(Boolean).join(" · ")));
        return row;
      }),
    );
    $("title").textContent = state.hasNotes ? notes.title : "Ask your notes";
    $("meta").textContent = state.hasNotes
      ? [chaptersLine(notes.chapters), plural(sources.length, "source"), plural(notes.pages, "page")].filter(Boolean).join(" · ")
      : "";
    $("summary").textContent = state.hasNotes ? notes.summary || "" : NO_NOTES;
    $("suggestions").replaceChildren(
      ...(notes.questions || []).map((question) => {
        const button = el("button", "", question);
        button.type = "button";
        button.addEventListener("click", () => ask({ text: question }));
        return button;
      }),
    );
    const chapter = $("quiz-chapter");
    const chosen = chapter.value;
    chapter.replaceChildren(new Option("Any chapter", ""), ...(notes.chapters || []).map((number) => new Option(`Chapter ${number}`, number)));
    chapter.value = [...chapter.options].some((option) => option.value === chosen) ? chosen : "";
    nameSummary();
    if (notes.problem) status(notes.problem, true);
  }

  // The summary button says what it will summarise: the chapter chosen above it, or everything.
  function nameSummary() {
    const chapter = $("quiz-chapter").value;
    $("summarise-label").textContent = chapter ? `Summarise chapter ${chapter}` : "Summarise the book";
  }

  function status(message, bad = false) {
    const line = $("notes-status");
    line.textContent = message;
    line.classList.toggle("bad", bad);
  }

  async function addSource(file) {
    if (!file || $("drop").classList.contains("busy")) return;
    if (file.size > state.maxUploadMb * 1024 * 1024) return status(`File is larger than ${state.maxUploadMb} MB.`, true);
    const body = new FormData();
    body.set("file", file);
    await addNotes("/api/notes", body, `Reading ${file.name}…`);
    $("file").value = "";
  }

  async function addNotes(path, body, hint) {
    const drop = $("drop");
    const into = state.notebook;
    body.set("language", state.language);
    body.set("notebook", into);
    drop.classList.add("busy");
    $("drop-hint").textContent = hint;
    status("");
    try {
      const added = await api(path, { method: "POST", body });
      showNotebooks(added.notebooks);       // a notebook with no name yet takes one from its first source
      if (into !== state.notebook) return;  // another notebook was opened meanwhile: this one's notes are not shown in it
      showNotes(added.notes);
      status(added.status);
    } catch (error) {
      if (into === state.notebook) status(error.message, true);
      else toast(error.message);
    } finally {
      drop.classList.remove("busy");
      $("drop-hint").textContent = DROP_HINT;
    }
  }

  // ---------- Notebooks: each has its own sources and its own conversations ----------

  const picker = $("notebook");
  const naming = $("notebook-name");
  const busy = () => state.waiting || $("drop").classList.contains("busy");

  function showNotebooks(notebooks) {
    if (notebooks) state.notebooks = notebooks;
    picker.replaceChildren(...state.notebooks.map((notebook) => new Option(notebook.title, notebook.id)));
    picker.value = state.notebook;
    picker.title = state.notebooks.find((notebook) => notebook.id === state.notebook)?.title || "";
    if (state.hasNotes && picker.title) $("title").textContent = picker.title;      // the heading is the notebook's name
  }

  // Open a notebook: its sources, its recents and its latest conversation replace what is on the page.
  async function openNotebook(id) {
    if (busy()) {
      picker.value = state.notebook;
      return toast("Wait for this to finish first.");
    }
    try {
      show(await stateOf(id));
      leaf.scrollTo({ top: 0 });
    } catch (error) {
      picker.value = state.notebook;
      toast(error.message);
    }
  }

  async function newNotebook() {
    if (busy()) return toast("Wait for this to finish first.");
    try {
      const made = await sendJson("/api/notebooks", {});
      state.notebooks = made.notebooks;
      await openNotebook(made.notebook);
    } catch (error) {
      toast(error.message);
    }
  }

  // The name is edited in place: the list gives way to a text field until Enter, Escape or a click elsewhere.
  function renameNotebook() {
    naming.value = state.notebooks.find((notebook) => notebook.id === state.notebook)?.title || "";
    picker.hidden = true;
    naming.hidden = false;
    naming.focus();
    naming.select();
  }

  async function finishRenaming(keep) {
    if (naming.hidden) return;
    naming.hidden = true;
    picker.hidden = false;
    const title = naming.value.trim();
    const before = state.notebooks.find((notebook) => notebook.id === state.notebook)?.title;
    if (!keep || !title || title === before) return;
    try {
      showNotebooks((await sendJson(`/api/notebooks/${state.notebook}/rename`, { title })).notebooks);
    } catch (error) {
      toast(error.message);
    }
  }

  async function deleteNotebook() {
    if (busy()) return toast("Wait for this to finish first.");
    const title = state.notebooks.find((notebook) => notebook.id === state.notebook)?.title || "this notebook";
    if (!confirm(`Delete "${title}" with its sources and conversations? This cannot be undone.`)) return;
    try {
      const left = await sendJson(`/api/notebooks/${state.notebook}/delete`, {});
      state.notebooks = left.notebooks;
      await openNotebook(left.notebook);
    } catch (error) {
      toast(error.message);
    }
  }

  // ---------- The conversation on the page ----------

  const toEnd = (smooth = true) => leaf.scrollTo({ top: leaf.scrollHeight, behavior: smooth && !calm ? "smooth" : "auto" });

  function question(words, spoken) {
    const node = el("div", "q");
    if (spoken) node.append(icon(ICONS.mic));
    node.append(el("p", "", words));
    return node;
  }

  function thinking() {
    const node = el("section", "a thinking");
    const dots = el("div", "text");
    dots.append(el("i"), el("i"), el("i"));
    node.append(dots);
    return node;
  }

  // A reply as paragraphs, with any formula drawn. A formula between $$ may run over several lines.
  function write(node, words) {
    const paragraphs = [];
    for (const [index, block] of words.split(/(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\])/).entries()) {
      if (index % 2) paragraphs.push(block.replace(/\s*\n\s*/g, " "));
      else paragraphs.push(...block.split(/\n+/).filter((line) => line.trim()));
    }
    node.replaceChildren(...paragraphs.map((paragraph) => el("p", "", paragraph)));
    if (window.renderMathInElement) window.renderMathInElement(node, { delimiters: MATH, throwOnError: false });
  }

  function fillAnswer(node, message, chat) {
    node.className = message.can_speak ? "a" : "a plain";
    const text = el("div", "text");
    write(text, message.text);
    const cite = el("p", "cite", message.cited.map((from) => `${from.source} · ${from.pages.length === 1 ? "page" : "pages"} ${from.pages.join(", ")}`).join("  ·  "));
    node.replaceChildren(text, cite, narration(message, chat));
  }

  function showChat(chat) {
    talk.replaceChildren();
    for (const message of chat.messages) {
      if (message.role === "user") talk.append(question(message.text, message.spoken));
      else {
        const node = el("section");
        fillAnswer(node, message, chat.id);
        talk.append(node);
      }
    }
    toEnd(false);
  }

  function showRecents() {
    $("no-recents").hidden = state.chats.length > 0;
    $("recents").replaceChildren(
      ...state.chats.map((chat) => {
        const row = el("li");
        const button = el("button", chat.id === state.chat ? "on" : "", chat.title || "New chat");
        button.type = "button";
        button.title = chat.title;
        button.addEventListener("click", () => openChat(chat.id));
        row.append(button);
        return row;
      }),
    );
  }

  async function openChat(id) {
    if (state.waiting || id === state.chat) return closeDrawers();
    try {
      const chat = await api(`/api/chats/${id}`);
      quiet();
      state.chat = chat.id;
      showChat(chat);
      showRecents();
    } catch (error) {
      toast(error.message);
    }
    closeDrawers();
  }

  function newChat() {
    if (state.waiting) return;
    quiet();
    state.chat = "";
    talk.replaceChildren();
    showRecents();
    leaf.scrollTo({ top: 0 });
    closeDrawers();
    field.focus();
  }

  function showDetails(details) {
    if (!details) return;
    const rows = [["Heard", details.heard], ["Language", details.language], ["Looked up", details.tool], ["Passages", details.passages],
                  ["Timings", details.timings], ["Tokens", details.tokens], ["Error", details.error]];
    $("details").replaceChildren(...rows.filter(([, value]) => value).flatMap(([name, value]) => [el("dt", "", name), el("dd", "", value)]));
  }

  // ---------- The narration under each answer ----------

  const narrator = new Audio();
  let narrating = null;       // the narration row the audio belongs to

  narrator.addEventListener("play", () => narrating?.classList.add("playing"));
  narrator.addEventListener("pause", () => narrating?.classList.remove("playing"));
  narrator.addEventListener("timeupdate", () => {
    if (!narrating) return;
    const length = narrator.duration || 0;
    narrating.querySelector(".fill").style.width = length ? `${(narrator.currentTime / length) * 100}%` : "0";
    narrating.querySelector(".time").textContent = `${clock(narrator.currentTime)} / ${clock(length)}`;
  });

  // Stop whatever is being read out: the made voice, or the browser's own.
  function quiet() {
    narrator.pause();
    window.speechSynthesis?.cancel();
  }

  // The browser's own voice for a language, if this browser has one. A Hindi voice can read Marathi's script.
  function browserVoice(language) {
    const voices = window.speechSynthesis?.getVoices() || [];
    for (const code of language === "mr" ? ["mr", "hi"] : [language]) {
      const voice = voices.find((candidate) => candidate.lang.toLowerCase().startsWith(code));
      if (voice) return voice;
    }
    return null;
  }

  // Read a reply with the browser's voice: free, used when the paid voice is over its budget or cannot be made.
  function sayWithBrowser(row, speech) {
    const voice = speech && browserVoice(speech.language);
    if (!voice) return false;
    quiet();
    const words = new SpeechSynthesisUtterance(speech.text);
    words.voice = voice;
    words.lang = voice.lang;
    words.rate = Number(row.dataset.rate || 1);
    words.onend = words.onerror = () => row.classList.remove("playing");
    narrating = row;
    delete row.dataset.voice;       // no made audio belongs to this row now
    row.dataset.browser = "yes";
    row.querySelector(".who").textContent = "Browser voice";
    row.classList.add("playing");
    window.speechSynthesis.speak(words);
    return true;
  }

  const voiceName = () => (state.voices.find((voice) => voice.id === state.voice)?.label || "").replace(/\s*\(.*\)/, "");

  function narration(message, chat) {
    const row = el("div", "narration");
    const play = el("button", "play");
    play.type = "button";
    play.setAttribute("aria-label", "Listen to this answer");
    play.append(icon(ICONS.play, "start"), icon(ICONS.pause, "pause"));
    const bars = el("span", "eq");
    bars.append(el("i"), el("i"), el("i"));
    const track = el("div", "track");
    track.append(el("div", "fill"));
    const speed = el("button", "speed", "1×");
    speed.type = "button";
    speed.setAttribute("aria-label", "Reading speed");
    row.append(play, bars, el("span", "who", voiceName()), track, el("span", "time", "0:00"), speed, actions(message, chat));
    row.listen = () => listen(row, message, chat);

    play.addEventListener("click", () => row.listen());
    track.addEventListener("click", (event) => {
      if (narrating !== row || !narrator.duration) return;
      const box = track.getBoundingClientRect();
      narrator.currentTime = ((event.clientX - box.left) / box.width) * narrator.duration;
    });
    speed.addEventListener("click", () => {
      const next = SPEEDS[(SPEEDS.indexOf(Number(row.dataset.rate || 1)) + 1) % SPEEDS.length];
      row.dataset.rate = next;
      speed.textContent = `${next}×`;
      if (narrating === row) narrator.playbackRate = next;
    });
    return row;
  }

  // Play an answer aloud. The voice is made the first time it is asked for; after that the same audio is replayed.
  async function listen(row, message, chat) {
    if (row.dataset.browser && window.speechSynthesis?.speaking) {
      row.classList.remove("playing");
      return window.speechSynthesis.cancel();
    }
    if (narrating === row && row.dataset.voice === state.voice && narrator.src) {
      return narrator.paused ? narrator.play() : narrator.pause();
    }
    if (row.classList.contains("loading")) return;
    const asked = state.asked;
    row.classList.add("loading");
    try {
      const made = await sendJson("/api/speak", { chat, key: message.key, voice: state.voice });
      showDetails(made.details);
      if (asked !== state.asked) return;      // a newer question has been asked meanwhile: stay quiet
      quiet();
      narrating = row;
      delete row.dataset.browser;
      row.dataset.voice = state.voice;
      row.querySelector(".who").textContent = voiceName();
      row.classList.add("ready");
      narrator.src = made.url;
      narrator.playbackRate = Number(row.dataset.rate || 1);
      await narrator.play();
    } catch (error) {
      showDetails(error.body?.details);
      // Over the budget, or the voice failed: the browser's own voice reads it if there is one for this language.
      const spoken = asked === state.asked && sayWithBrowser(row, error.body?.speech);
      if (error.name !== "NotAllowedError") toast(spoken ? `${error.message} Your browser's own voice is reading this one.` : error.message);
    } finally {
      row.classList.remove("loading");
    }
  }

  function actions(message, chat) {
    const group = el("span", "acts");
    const button = (shape, label, onClick) => {
      const node = el("button");
      node.type = "button";
      node.setAttribute("aria-label", label);
      node.title = label;
      node.append(icon(shape));
      node.addEventListener("click", () => onClick(node));
      group.append(node);
      return node;
    };
    button(ICONS.copy, "Copy", async () => {
      await navigator.clipboard.writeText(message.text).catch(() => {});
      toast("Copied.");
    });
    if (message.can_rate) {
      const rate = async (liked) => {
        const { saved } = await sendJson("/api/feedback", { chat, key: message.key, liked }).catch(() => ({ saved: false }));
        if (saved) {
          up.classList.toggle("on", liked);
          down.classList.toggle("on", !liked);
        }
        toast(saved ? "Thanks for the feedback." : "Could not save your feedback.");
      };
      const up = button(ICONS.up, "Helpful", () => rate(true));
      const down = button(ICONS.down, "Not helpful", () => rate(false));
      up.classList.toggle("on", message.liked === true);
      down.classList.toggle("on", message.liked === false);
    }
    return group;
  }

  // ---------- Asking ----------

  // The question leaves the composer and lands on the page as a message, the arrow leading the way.
  function fly(words, spoken) {
    const landed = question(words, spoken);
    talk.append(landed);
    toEnd(false);
    if (calm) return Promise.resolve();
    const from = composer.getBoundingClientRect();
    const to = landed.getBoundingClientRect();
    const start = send.getBoundingClientRect();
    landed.style.animation = "none";
    landed.style.visibility = "hidden";

    const ghost = landed.cloneNode(true);
    ghost.classList.add("ghost");
    Object.assign(ghost.style, { visibility: "", left: `${to.left}px`, top: `${to.top}px`, width: `${to.width}px` });
    const arrow = el("span", "ghost-arrow");
    arrow.append(icon(ICONS.arrow));
    Object.assign(arrow.style, { left: `${start.left}px`, top: `${start.top}px` });
    document.body.append(ghost, arrow);
    send.classList.add("launched");

    const dx = from.left + 14 - to.left;
    const dy = from.top + (from.height - to.height) / 2 - to.top;
    const ax = to.right - 20 - start.left;
    const ay = to.top + to.height / 2 - 15 - start.top;
    arrow.animate(
      [{ transform: "none", opacity: 1 }, { transform: `translate(${ax}px, ${ay}px)`, opacity: 1, offset: 0.75 }, { transform: `translate(${ax}px, ${ay}px) scale(0.3)`, opacity: 0 }],
      { duration: 560, easing: "cubic-bezier(0.3, 0.7, 0.2, 1)", fill: "both" },
    );
    const flight = ghost.animate(
      [{ transform: `translate(${dx}px, ${dy}px) scale(0.97)`, opacity: 0.55 }, { opacity: 1, offset: 0.3 }, { transform: "none", opacity: 1 }],
      { duration: 660, delay: 50, easing: SPRING, fill: "both" },
    );
    return flight.finished.then(() => {
      ghost.remove();
      arrow.remove();
      landed.style.visibility = "";
      landed.animate([{ transform: "scale(1.03)" }, { transform: "none" }], { duration: 240, easing: SPRING });
      setTimeout(() => send.classList.remove("launched"), 160);
    });
  }

  // "idle", "listening" or "transcribing": what the composer shows in place of the text field.
  function composerShows(mode, label = "") {
    composer.classList.remove("listening", "transcribing");
    if (mode !== "idle") composer.classList.add(mode);
    $("hearing").textContent = label;
    mic.setAttribute("aria-pressed", String(mode === "listening"));
  }

  async function ask({ text = "", clip = null, action = "", chapter = "" }) {
    if (state.waiting) return;
    state.waiting = true;
    composer.classList.add("waiting");
    const turn = ++state.asked;
    quiet();
    let pending = null;
    const place = async (words) => {
      composerShows("idle");
      await fly(words, clip !== null);
      pending = thinking();
      talk.append(pending);
      toEnd();
    };
    const body = new FormData();
    const asking = { chat: state.chat, notebook: state.notebook, language: state.language, text, action, chapter };
    for (const [name, value] of Object.entries(asking)) body.set(name, value);
    if (clip) body.set("audio", clip, "question");
    try {
      if (text) await place(text);
      const response = await fetch("/api/ask", { method: "POST", body });
      if (!response.ok) {
        const refusal = await response.json().catch(() => ({}));
        if (response.status === 401) $("lock").hidden = false;
        throw new Error((typeof refusal.detail === "string" && refusal.detail) || refusal.message || refusal.error || "Something went wrong. Please try again.");
      }
      for await (const event of events(response)) {
        if (event.event === "heard" && !pending) await place(event.text);
        if (event.event === "error") {
          pending?.remove();
          toast(event.message);
        }
        if (event.event === "reply") {
          state.chat = event.chat.id;
          state.chats = [event.chat, ...state.chats.filter((chat) => chat.id !== event.chat.id)];
          showRecents();
          if (!pending) talk.append((pending = thinking()));
          fillAnswer(pending, event.message, state.chat);
          showDetails(event.details);
          toEnd();
          if (event.speak && turn === state.asked) pending.querySelector(".narration").listen();
        }
      }
    } catch (error) {
      pending?.remove();
      toast(error.message);
    } finally {
      state.waiting = false;
      composer.classList.remove("waiting");
      composerShows("idle");
    }
  }

  function sendTyped() {
    const text = field.value.trim();
    if (!text || state.waiting) return field.focus();
    if (text.length > state.maxQuestionChars) return toast("That question is too long. Please ask it in fewer words.");
    field.value = "";
    fit();
    ask({ text });
  }

  function fit() {
    field.style.height = "auto";
    field.style.height = `${field.scrollHeight}px`;
    composer.classList.toggle("has-text", field.value.trim() !== "");
  }

  // ---------- The microphone ----------

  let recorder = null;
  let stream = null;
  let chunks = [];
  let sounds = null;          // the AudioContext for the cues and the live waveform
  let analyser = null;
  let levels = [];
  let frame = 0;
  let cutOff = 0;
  let starting = false;

  // A short, soft pair of notes: rising when listening starts, falling when it ends.
  function cue(rising) {
    const now = sounds.currentTime;
    for (const [index, pitch] of (rising ? [587, 784] : [659, 494]).entries()) {
      const tone = sounds.createOscillator();
      const volume = sounds.createGain();
      tone.frequency.value = pitch;
      volume.gain.setValueAtTime(0, now + index * 0.07);
      volume.gain.linearRampToValueAtTime(0.05, now + index * 0.07 + 0.015);
      volume.gain.exponentialRampToValueAtTime(0.0001, now + index * 0.07 + 0.11);
      tone.connect(volume).connect(sounds.destination);
      tone.start(now + index * 0.07);
      tone.stop(now + index * 0.07 + 0.12);
    }
  }

  // The live waveform: one bar for each twentieth of a second, newest at the microphone's side.
  function drawWave() {
    const canvas = $("wave");
    const sharp = window.devicePixelRatio || 1;
    if (canvas.width !== canvas.clientWidth * sharp) {
      canvas.width = canvas.clientWidth * sharp;
      canvas.height = canvas.clientHeight * sharp;
    }
    const samples = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(samples);
    const loudness = Math.sqrt(samples.reduce((sum, value) => sum + (value - 128) ** 2, 0) / samples.length);
    if (frame % 3 === 0) levels.push(Math.min(1, loudness / 38));
    const step = 5 * sharp;
    levels = levels.slice(-Math.floor(canvas.width / step));
    const pen = canvas.getContext("2d");
    pen.clearRect(0, 0, canvas.width, canvas.height);
    pen.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent");
    levels.forEach((level, index) => {
      const height = Math.max(2 * sharp, level * canvas.height * 0.94);
      const x = canvas.width - (levels.length - index) * step;
      pen.beginPath();
      pen.roundRect(x, (canvas.height - height) / 2, 2.4 * sharp, height, sharp);
      pen.fill();
    });
    frame = requestAnimationFrame(drawWave);
  }

  async function startListening() {
    starting = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      quiet();        // the last reply must not be recorded
      sounds = sounds || new AudioContext();
      await sounds.resume();
      cue(true);
      analyser = sounds.createAnalyser();
      analyser.fftSize = 512;
      sounds.createMediaStreamSource(stream).connect(analyser);
      levels = [];
      composerShows("listening", "Listening…");
      drawWave();
      await pause(150);       // the cue is over before the recording begins
      chunks = [];
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (event) => chunks.push(event.data);
      recorder.start();
      cutOff = setTimeout(stopListening, state.maxClipMs);     // the longest clip the app accepts is sent as it is
    } catch (error) {
      composerShows("idle");
      toast(`The microphone could not be used: ${error.message}`);
    } finally {
      starting = false;
    }
  }

  function stopListening() {
    if (!recorder) return;
    const finished = recorder;
    recorder = null;
    clearTimeout(cutOff);
    cancelAnimationFrame(frame);        // the waveform freezes as it was
    finished.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());      // releases the microphone
      ask({ clip: new Blob(chunks, { type: finished.mimeType }) });
    };
    finished.stop();
    cue(false);
    composerShows("transcribing", "Writing it down…");
  }

  // ---------- The side panels on small screens ----------

  const closeDrawers = () => app.classList.remove("shelf-open", "studio-open");

  // ---------- Wiring ----------

  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    sendTyped();
  });
  field.addEventListener("input", fit);
  field.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendTyped();
    }
  });
  mic.addEventListener("click", () => {
    if (starting || state.waiting) return;
    if (recorder) stopListening();
    else startListening();
  });

  $("file").addEventListener("change", (event) => addSource(event.target.files[0]));
  const drop = $("drop");
  drop.addEventListener("keydown", (event) => (event.key === "Enter" || event.key === " ") && $("file").click());
  for (const name of ["dragenter", "dragover"]) drop.addEventListener(name, (event) => (event.preventDefault(), drop.classList.add("over")));
  for (const name of ["dragleave", "drop"]) drop.addEventListener(name, (event) => (event.preventDefault(), drop.classList.remove("over")));
  drop.addEventListener("drop", (event) => addSource(event.dataTransfer.files[0]));
  $("sample").addEventListener("click", () => addNotes("/api/notes/sample", new FormData(), "Adding the sample notes…"));

  $("notice-ok").addEventListener("click", () => {
    localStorage.setItem("notice-read", "yes");
    $("first-notice").hidden = true;
  });
  $("forget").addEventListener("click", async () => {
    if (state.waiting || !confirm("Remove all your notebooks, uploaded files and conversations from this app? This cannot be undone.")) return;
    try {
      await sendJson("/api/forget", {});
      localStorage.removeItem("notebook");
      location.reload();
    } catch (error) {
      toast(error.message);
    }
  });
  picker.addEventListener("change", () => openNotebook(picker.value));
  $("new-notebook").addEventListener("click", newNotebook);
  $("rename-notebook").addEventListener("click", renameNotebook);
  $("delete-notebook").addEventListener("click", deleteNotebook);
  naming.addEventListener("blur", () => finishRenaming(true));
  naming.addEventListener("keydown", (event) => {
    if (event.key === "Enter") finishRenaming(true);
    if (event.key === "Escape") finishRenaming(false);
  });
  $("new-chat").addEventListener("click", newChat);
  $("quiz").addEventListener("click", () => (closeDrawers(), ask({ action: "quiz", chapter: $("quiz-chapter").value })));
  $("summarise").addEventListener("click", () => (closeDrawers(), ask({ action: "summary", chapter: $("quiz-chapter").value })));
  $("quiz-chapter").addEventListener("change", nameSummary);

  for (const button of $("language").children) {
    button.setAttribute("aria-checked", String(button.dataset.value === state.language));
    button.addEventListener("click", async () => {
      if (button.dataset.value === state.language) return;
      state.language = button.dataset.value;
      localStorage.setItem("language", state.language);
      for (const other of $("language").children) other.setAttribute("aria-checked", String(other === button));
      // The questions to try are written in the language the student speaks.
      if (state.hasNotes) showNotes((await stateOf(state.notebook)).notes);
    });
  }
  $("voice").addEventListener("change", (event) => {
    state.voice = event.target.value;
    localStorage.setItem("voice", state.voice);
  });

  $("open-shelf").addEventListener("click", () => app.classList.toggle("shelf-open"));
  $("open-studio").addEventListener("click", () => app.classList.toggle("studio-open"));
  $("backdrop").addEventListener("click", closeDrawers);

  $("lock-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await sendJson("/api/access", { code: $("code").value });
      $("lock").hidden = true;
      $("code").value = "";
      open();
    } catch (error) {
      $("lock-status").textContent = error.message;
    }
  });
  window.speechSynthesis?.getVoices();     // browsers load their voices only after being asked once

  // Put one notebook on the page: its sources, its recents and its latest conversation.
  function show(opened) {
    quiet();
    state.notebook = opened.notebook;
    localStorage.setItem("notebook", state.notebook);
    showNotebooks(opened.notebooks);
    status("");
    showNotes(opened.notes);
    state.chats = opened.chats;
    state.chat = opened.chat ? opened.chat.id : "";
    if (opened.chat) showChat(opened.chat);
    else talk.replaceChildren();
    showRecents();
    $("details").replaceChildren(el("dt", "", "Nothing asked yet."));
  }

  async function open() {
    try {
      const first = await stateOf(state.notebook);
      if (first.locked) {
        $("lock").hidden = false;
        return $("code").focus();
      }
      state.voices = first.voices;
      state.voice = first.voices.some((voice) => voice.id === state.voice) ? state.voice : first.voice;
      state.maxClipMs = first.max_clip_s * 1000;
      state.maxUploadMb = first.max_upload_mb;
      state.maxQuestionChars = first.max_question_chars;
      $("voice").replaceChildren(...first.voices.map((voice) => new Option(voice.label, voice.id)));
      $("voice").value = state.voice;
      $("notice").textContent = first.notice;
      if (!localStorage.getItem("notice-read")) {
        $("first-notice").firstChild.textContent = first.notice;
        $("first-notice").hidden = false;
      }
      show(first);
    } catch (error) {
      toast("The app is not answering. Check that it is still running, then reload this page.");
    }
  }
  open();
})();
