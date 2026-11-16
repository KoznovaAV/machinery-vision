
"use strict";
const byId = (id) => document.getElementById(id);
const form = byId("review");
const button = byId("predict");
const drop = byId("drop");
const picker = byId("photo");
let file = null, second = null, previewUrl = null;
let revision = 0, photoRevision = 0, saving = false;
const familyNames = {chtz:"ЧТЗ",johndeere:"John Deere",kirovets:"Кировец",mtz_belarus:"МТЗ / Беларус"};
const modelNames = {mtz_82:"МТЗ‑82",mtz_1221:"МТЗ‑1221"};
const percent = new Intl.NumberFormat("ru",{style:"percent",maximumFractionDigits:1});
const dirty = (record) => {
  const label = record.p_dirty >= record.threshold ? "Грязь видна" : "Грязь не видна";
  return label + " · " + percent.format(record.p_dirty);
};
function status(id, message, kind="") {
  const element = byId(id);
  element.className = "status " + kind;
  element.textContent = message;
}
function updateSave() {
  byId("save").disabled = !file || saving || ["model","mud","condition"].some(key => !byId(key).value);
}
function choose(next) {
  if (!next) return;
  const format = /\.(jpe?g|png|webp)$/i.test(next.name);
  if (!format || next.size > 12 * 1024 * 1024) {
    status("prediction-status","Выберите JPEG, PNG или WebP размером до 12 МБ.","error");
    return;
  }
  revision += 1;
  photoRevision += 1;
  file = next;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  byId("preview").src = previewUrl;
  byId("preview").hidden = false;
  byId("placeholder").hidden = true;
  second = null;
  byId("photo-second-name").textContent = "";
  byId("second-actions").hidden = false;
  byId("remove-second").hidden = true;
  resetPrediction();
  byId("before-results").hidden = false;
  byId("photo-name").textContent = file.name;
  button.disabled = false;
  button.textContent = "Распознать";
  form.reset();
  saving = false;
  byId("review-fields").disabled = false;
  updateSave();
  status("prediction-status","");
  status("review-status","");
}
function resetPrediction() {
  byId("scores").hidden = true;
  byId("decision").hidden = true;
  byId("decision-tip").hidden = true;
  byId("before-results").hidden = false;
  button.disabled = !file;
  button.textContent = "Распознать";
  status("prediction-status","");
}
picker.addEventListener("change", () => {
  choose(picker.files[0]);
  picker.value = "";
});
const secondPicker = byId("photo-second");
secondPicker.addEventListener("change", () => {
  const next = secondPicker.files[0];
  secondPicker.value = "";
  if (!next) return;
  if (!/\.(jpe?g|png|webp)$/i.test(next.name) || next.size > 12 * 1024 * 1024) {
    status("prediction-status","Выберите JPEG, PNG или WebP размером до 12 МБ.","error");
    return;
  }
  second = next;
  revision += 1;
  byId("photo-second-name").textContent = "Второй ракурс · " + next.name;
  byId("remove-second").hidden = false;
  resetPrediction();
});
byId("remove-second").addEventListener("click", () => {
  second = null;
  revision += 1;
  byId("photo-second-name").textContent = "";
  byId("remove-second").hidden = true;
  resetPrediction();
});
for (const event of ["dragenter","dragover"]) {
  drop.addEventListener(event,(e) => { e.preventDefault(); drop.classList.add("active"); });
}
for (const event of ["dragleave","drop"]) {
  drop.addEventListener(event,(e) => { e.preventDefault(); drop.classList.remove("active"); });
}
drop.addEventListener("drop",(event) => choose(event.dataTransfer.files[0]));
form.addEventListener("change", updateSave);
async function post(url, payload) {
  const response = await fetch(url,{method:"POST",body:payload});
  const value = await response.json();
  if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "Ошибка локального сервера.");
  return value;
}
button.addEventListener("click", async () => {
  if (!file) return;
  const current = revision;
  const payload = new FormData();
  payload.append("file",file);
  if (second) payload.append("second",second);
  button.disabled = true;
  button.textContent = "Распознаю…";
  status("prediction-status","Первый запуск может занять несколько секунд.");
  try {
    const v = await post("/api/predict",payload);
    if (current !== revision) return;
    const decision = v.decision;
    const family = familyNames[decision.family] ?? "";
    const labels = {
      model: modelNames[decision.model],
      family_only: family,
      outside_classes: "Возможно, техника вне известных классов",
      manual_review: "Нужна ручная проверка",
      unavailable: "Фильтр пока недоступен",
    };
    byId("decision").hidden = false;
    byId("decision").textContent = labels[decision.status];
    byId("decision").classList.toggle("decision-review", !["model","family_only"].includes(decision.status));
    const tip = byId("decision-tip");
    tip.hidden = !["outside_classes","manual_review"].includes(decision.status);
    tip.textContent = second
      ? "Уточните марку и модель по маркировке на машине."
      : "Если модель важна, добавьте фото той же машины сбоку под углом или крупный план маркировки для ручной проверки.";
    const rows = [];
    for (const [index, view] of [v, v.second].entries()) {
      if (!view) continue;
      const suffix = v.second ? " · фото " + (index + 1) : "";
      rows.push(["Семейство · v3" + suffix,(familyNames[view.family] ?? view.family)+" · "+percent.format(view.family_score)]);
      for (const [key, label] of [
        ["mtz","Модель МТЗ · базовая"],["mtz_new","Модель МТЗ · пилот"],
        ["mtz_weighted","Модель МТЗ · пилот с весами"],["mtz_expanded","Модель МТЗ · 100 новых фото"],
        ["mtz_expanded_group_balanced","Модель МТЗ · 100 фото с весами групп"],
      ]) {
        const m = view.mtz[key];
        if (m) rows.push([label + suffix,modelNames[m.model]+" · "+percent.format(Math.max(m.p82,1-m.p82))]);
      }
      for (const [key, label] of [["family","Фильтр семейств"],["mtz","Фильтр моделей МТЗ"]]) {
        const candidate = view.filter[key]?.linear;
        if (candidate && (key === "family" || view.family === "mtz_belarus")) {
          rows.push([label + suffix,candidate.accepted ? "Прошёл" : "Не прошёл"]);
        }
      }
      rows.push(["Грязь · v3" + suffix,dirty(view.v3)],["Грязь · кандидат" + suffix,dirty(view.candidate)]);
    }
    const list = byId("results");
    list.replaceChildren(...rows.map(([label,value]) => {
      const row = document.createElement("div"), dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = label; dd.textContent = value; row.append(dt,dd); return row;
    }));
    byId("scores").hidden = false;
    byId("scores").open = decision.status === "unavailable";
    byId("before-results").hidden = true;
    status("prediction-status","");
  } catch (error) {
    if (current === revision) status("prediction-status",error.message,"error");
  } finally {
    if (current === revision) { button.disabled = false; button.textContent = "Распознать"; }
  }
});
form.addEventListener("submit",async (event) => {
  event.preventDefault();
  if (!file || saving) return;
  const current = photoRevision;
  const labels = Object.fromEntries(new FormData(form));
  const payload = new FormData();
  payload.append("file",file); payload.append("labels",JSON.stringify(labels));
  saving = true; updateSave(); status("review-status","Сохраняю…");
  try {
    await post("/api/review",payload);
    if (current === photoRevision) status("review-status","Оценка сохранена локально.","success");
  } catch (error) {
    if (current === photoRevision) status("review-status",error.message,"error");
  } finally {
    if (current === photoRevision) { saving = false; updateSave(); }
  }
});
