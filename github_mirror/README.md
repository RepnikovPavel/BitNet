# github_mirror — зеркало issues и pull requests microsoft/BitNet

Полный снапшот всех issues и pull requests оригинального репозитория
[microsoft/BitNet](https://github.com/microsoft/BitNet) (все состояния:
open/closed/merged), включая комментарии и review-комментарии к PR.

Снят инструментом [ghdump](https://github.com/RepnikovPavel/ghdump):

```sh
ghdump microsoft/BitNet github_mirror
```

Структура:

- `index.json` — сводка: количества и краткие метаданные каждого элемента;
- `issues/NNNNNN.json` / `issues/NNNNNN.md` — каждый issue в сыром виде
  (ответ API + `_comments`) и в читаемом Markdown;
- `pull_requests/NNNNNN.json` / `pull_requests/NNNNNN.md` — то же для PR
  (дополнительно `_review_comments`).

Дата снятия снапшота — в поле `fetched_at` файла `index.json`.
