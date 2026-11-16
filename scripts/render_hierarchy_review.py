"""Render human-readable hierarchy outputs with explicit uncertainty."""

import html
import json
from pathlib import Path
from urllib.parse import quote

from src.config.machinery_taxonomy import CATALOG, LABELS, TYPE_NAMES


def render(data, asset_root=None):
    def esc(value):
        return html.escape(str(value or ""))

    def asset(value):
        path = Path(value).resolve()
        if asset_root is not None:
            return "/artifacts/" + quote(path.relative_to(Path(asset_root).resolve()).as_posix())
        return path.as_uri()

    cards = []
    for item in data.get("objects", []):
        family = item.get("family_name") or item.get("family_candidate_name") or "Не определено"
        state = (
            "Семейство определено; точная модель неясна"
            if item.get("status") == "family_only"
            else "Нужна проверка человеком"
        )
        fine = item.get("model_candidate_name") or CATALOG["models"].get(
            item.get("model_candidate"), {}
        ).get("name", "Не определена")
        candidate = LABELS.get(item.get("label_candidate"), {}).get("name", "Не определён")
        kind = (
            "Требуется уточнение"
            if item.get("type_conflict")
            else TYPE_NAMES.get(item.get("vehicle_type_candidate"), "Не определён")
        )
        family_title = "Семейство" if item.get("family") else "Семейство — кандидат"
        details = item.get("details", {}).get("front_axle")
        condition = {
            "clean": "Видимой грязи не обнаружено",
            "dirty": "Есть признаки загрязнения",
        }.get(item.get("condition"), "Состояние не удалось оценить")
        condition_reason = item.get("condition_reason")
        condition_note = f"<small>{esc(condition_reason)}</small>" if condition_reason else ""
        weather = ""
        names = {"snow": "Снег в кадре", "rain": "Дождь в кадре", "wet": "Влажность поверхности"}
        for key, title in names.items():
            attribute = item.get("condition_attributes", {}).get(key)
            if attribute:
                weather += f"<dt>{esc(title)}</dt><dd>{esc(attribute['text'])}</dd>"
        part = ""
        if isinstance(details, dict) and details.get("image"):
            part = (
                f'<figure><img src="{esc(asset(details["image"]))}">'
                "<figcaption>Область для проверки переднего моста; "
                "тип моста не подтверждён</figcaption></figure>"
            )
        pictures = "".join("<li>" + esc(x) + "</li>" for x in item.get("requested_photos", []))
        cards.append(
            f"""<article>
<h2>Машина {int(item.get('object_id',0))+1}</h2>
<p class="status">{esc(state)}</p>
<img src="{esc(asset(item['crop']))}" alt="Выделенная машина">
<dl>
<dt>Тип — кандидат</dt><dd>{esc(kind)}</dd>
<dt>{esc(family_title)}</dt><dd>{esc(family)}</dd>
<dt>Группа / серия — кандидат</dt><dd>{esc(candidate)}</dd>
<dt>Точная модель — кандидат</dt><dd>{esc(fine)}; не подтверждена</dd>
<dt>Видимая маркировка</dt>
<dd>{esc(item.get('visible_model_marking') or 'Не подтверждена')}</dd>
<dt>Загрязнение по фото</dt><dd>{esc(condition)}{condition_note}</dd>{weather}
</dl>
<p>{esc(item.get('reason'))}</p>{part}
<p>Для уточнения:</p><ul>{pictures}</ul>
</article>"""
        )
    content = "".join(cards)
    if not content and any(r.get("reason") == "semantic_trailer" for r in data.get("ignored_regions", [])):
        content = "<p>Обнаружен прицеп. Он не считается отдельной самоходной машиной.</p>"
    if not content and any(r.get("reason") == "attachment" for r in data.get("ignored_regions", [])):
        content = "<p>Обнаружено прицепное или навесное оборудование. Самоходная техника не подтверждена.</p>"
    if not content:
        content = (
            "<p>Загрузите фотографию для проверки.</p>"
            if data.get("initial")
            else "<p>Поддерживаемая техника не найдена. Попробуйте более крупный общий план.</p>"
        )
    for region in data.get("uncertain_regions", []):
        picture = (f'<img src="{esc(asset(region["crop"]))}" alt="Неподтверждённая область">'
                   if region.get("crop") else "")
        content += ("<aside class=\"notice\"><h2>Неподтверждённая область</h2>"
                    + picture + "<p>Возможно, это прицеп, орудие или фрагмент машины. "
                    "Не считаем отдельной самоходной машиной без проверки человеком.</p></aside>")
    return (
        """<!doctype html><html lang="ru">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Проверка распознавания техники</title>
<style>
body{font:16px system-ui;background:#f3f5f7;color:#1c2733;margin:0}
main{max-width:1100px;margin:auto;padding:24px}h1{font-size:27px}
.notice{padding:16px;background:#fff4cf;border-radius:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}
article,form{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin-top:20px}
img{width:100%;max-height:380px;object-fit:contain}
dt{color:#566476;margin-top:10px}dd{margin:3px 0;font-weight:600}
dd small{display:block;font-weight:400;color:#566476}
.status{color:#704700}figcaption{font-size:13px}button{padding:10px 18px}
</style><main><h1>Проверка распознавания техники</h1>
<p class="notice">Экспериментальный режим. Проверяйте кандидатов и маркировку.
Оценки видимого состояния могут потребовать уточняющего фото. Проверяйте результат перед использованием.</p>
<form action="/predict" method="post" enctype="multipart/form-data">
<label>Фото техники
<input name="file" type="file" accept="image/jpeg,image/png,image/webp" required></label>
<button>Проверить фото</button></form>
<section class="grid">"""
        + content
        + "</section></main></html>"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    args = parser.parse_args()
    data = json.loads(args.prediction.read_text(encoding="utf-8"))
    path = args.prediction.with_suffix(".html")
    path.write_text(render(data), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
