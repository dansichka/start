# Контракт с kernel

Компаньон на Mac ничего не знает об устройстве kernel. Он говорит с ним через шесть операций. Серверная половина для стороны kernel уже написана: вам остаётся подключить к ней свои данные, а весь остальной код kernel не меняется и никуда не уходит.

## Что лежит в серверной половине

Файлы из `src/call_companion/`, которые копируются на сторону kernel (Python 3.10+, проверено на 3.10.18):

| Файл | Роль | Зависимости |
|---|---|---|
| `__init__.py`, `models.py` | схемы данных | pydantic ≥ 2.7 |
| `util.py`, `bm25.py`, `redact.py` | поиск по основам слов, вычищение секретов | стандартная библиотека |
| `kernel_api.py` | `DataSource`, `dispatch`, `run_cli`, `write_pack`, `read_pack`, эталон `MemoryDataSource` | pydantic |
| `kernel_server.py` | HTTP: `create_app`, `serve` | fastapi, uvicorn |

Тест `test_kernel_side_modules_import_without_the_rest_of_the_package` следит, чтобы эти модули не тянули numpy, anthropic и прочее.

## Операции

Одна и та же логика обслуживает HTTP и CLI (`dispatch`), поэтому транспорты не расходятся.

| Операция | HTTP | CLI (`<команда> <op>`, JSON в stdin) | Запрос | Ответ |
|---|---|---|---|---|
| `health` | `GET /v1/health` | `health` | — | `{"ok": true, "version": "…", "items": 123}` |
| `search` | `POST /v1/search` | `search` | `{"query", "k", "filters"}` | `{"items": [ContextItem]}` |
| `brief` | `POST /v1/brief` | `brief` | `{"meeting": Meeting, "max_sensitivity": "work"}` | `Brief` |
| `upcoming` | `GET /v1/meetings/upcoming?within_min=30` | `upcoming` | `{"within_min": 30}` | `{"meetings": [Meeting]}` |
| `note` | `POST /v1/notes` | `note` | `Note` | `{"id": "…"}` |
| `action` | `POST /v1/actions` | `action` | `ProposedAction` | `{"id": "…"}` |

Правила сервера: токен `Authorization: Bearer …` обязателен (сервис не стартует без него), сравнение за постоянное время; тело до 1 МБ; `k` не больше 20; документация OpenAPI отключена; ответ всегда фильтруется по `max_sensitivity` и никогда не содержит `secret`, даже если `DataSource` вернул лишнее. `action` вызывается только после того, как человек подтвердил действие на Mac.

## Схемы

**ContextItem** — одна единица знания.

| Поле | Тип | Смысл |
|---|---|---|
| `id` | строка | стабильный, уникальный: `<источник>:<id в источнике>` |
| `source` | строка | `telegram`, `calendar`, `github`, `note`, `drive`, `kernel`, … |
| `kind` | строка | `message`, `note`, `event`, `pr`, `doc`; особые: `open_loop` (обещание пользователя), `talking_point` (заготовленный тезис) |
| `title`, `text` | строки | заголовок и содержание (до ~1000 символов), без секретов |
| `ts` | ISO 8601 | время (без пояса считается UTC) |
| `url` | строка или null | ссылка на оригинал |
| `entities` | список строк | люди, компании, проекты из материала: по ним связываются участники и упоминания |
| `sensitivity` | `public` / `work` / `private` / `secret` | неизвестное значение читается как `secret` |
| `score` | число | релевантность, ставит сервер поиска; сравнивается только внутри одного ответа |

**Meeting**: `id`, `title`, `start`, `end`, `attendees` (`name`, `email`, `org`, `aliases`), `agenda`, `platform`, `mode` (`internal` / `client` / `interview` / `public`, необязательно).
**Brief**: `meeting`, `items`, `entities` (`name`, `kind`, `aliases`, `summary`), `talking_points`, `open_loops`, `generated_at`.
**Note**: `text`, `tags`, `entities`, `meeting_id`, `source` (`call-companion`), `untrusted` (всегда `true`: текст вышел из звонка и не проверен человеком).
**ProposedAction**: `id`, `capability` (в kernel приходит только `kernel.task`), `title`, `payload`, `rationale`, `origin`. `id` служит ключом идемпотентности.
**SearchFilters**: `sources`, `since`, `entities`, `max_sensitivity`.

Неизвестные поля клиент игнорирует, так что kernel может добавлять свои.

## Метки чувствительности

Решает владелец kernel; ниже разумная отправная точка, при сомнении берите более строгую метку.

| Что | Метка |
|---|---|
| Публичные документы, ваши опубликованные тексты | `public` |
| Рабочие заметки, задачи, рабочие чаты, PR, события календаря | `work` |
| Личные чаты, личные заметки, переговорные позиции | `private` |
| Платежи, доходы, токены, пароли, всё из конфигов с секретами | `secret` (в выгрузку не попадает) |

Режим звонка задаёт потолок: `internal` до `private`, `client` и `interview` до `work`, `public` только `public`. Чтобы в собеседовании показывались ваши заметки о подготовке, разметьте их `work` или расширьте потолок в `[policy.mode_limits]`.

## Пакет контекста (работает без ПК)

Файл `pack.jsonl`: по объекту на строку, `{"type": "item" | "entity" | "meeting", …поля}` (без `type` читается как `item`). Пишите его только через `write_pack(path, items, entities, meetings)`: запись атомарная, синхронизатор не подхватит половину файла. Дальше любым синхронизатором (Syncthing, облачная папка, `scp`) на Mac в `~/.callcompanion/pack.jsonl`. Заметки, которые Mac не смог отправить, лежат рядом в `pack.jsonl.outbox.jsonl` (`type: "note"`), их kernel может забрать при следующей синхронизации. Проверка: `callc pack-info pack.jsonl` (код возврата 1, если есть битые строки).

## Что реализовать в kernel

```python
from call_companion.kernel_api import run_cli, write_pack
from call_companion.models import Brief, ContextItem, Meeting, Note, ProposedAction, SearchFilters

class KernelDataSource:
    def stats(self) -> dict: ...                                   # {"version": "...", "items": N}
    def search(self, query: str, k: int, filters: SearchFilters) -> list[ContextItem]: ...
    def brief(self, meeting: Meeting, max_sensitivity: str) -> Brief: ...
    def upcoming(self, within_min: int) -> list[Meeting]: ...      # из календарного сборщика
    def save_note(self, note: Note) -> str: ...                    # в заметки kernel с пометкой «не проверено»
    def run_action(self, action: ProposedAction) -> str: ...       # только kernel.task, остальное — исключение
```

Для `brief` можно взять эталон: `build_brief(index, items, entities, meeting, max_sensitivity)` из `kernel_api.py`, построив индекс по свежим материалам. Подключение:

```python
# kernel companion <op>        — тот же код для CLI (ssh pc kernel companion search < запрос.json)
raise SystemExit(run_cli(KernelDataSource(), [op]))

# kernel companion serve       — HTTP на адресе Tailscale
from call_companion.kernel_server import serve
serve(KernelDataSource(), token=os.environ["KERNEL_COMPANION_TOKEN"], host=tailscale_ip, port=8765)

# kernel companion export      — пакет для офлайна (раз в 30 минут из воркера)
write_pack(out_path, items, entities, meetings)
```

На ПК с Windows откройте порт только на интерфейсе Tailscale. Токен храните в переменной окружения.

## Как проверить

1. `callc pack-info pack.jsonl` показывает счётчики по источникам и меткам, пропущенных строк 0.
2. Без вашего kernel проверьте сеть на демо-данных: на ПК `callc demo-kernel --pack examples/sample_kernel_pack.jsonl --host <ip tailscale>`, на Mac в конфиге `kernel.backends = ["http"]`, затем `callc doctor` и `callc brief --title Тест --attendee "Max Weber"`.
3. Тесты на стороне kernel по образцу `tests/test_kernel_contract.py`: клиент против `create_app(KernelDataSource(), token)`: `secret` не возвращается никогда, `max_sensitivity` соблюдается, заметка сохраняется с `untrusted`, неверный токен даёт 401.

## Задание для Claude Code в папке kernel

Скопируйте целиком в локальную сессию, открытую в папке kernel.

```text
Ты работаешь в репозитории kernel на этом компьютере. Задача: добавить модуль kernel/companion/, через который внешний
«компаньон для созвонов» на MacBook читает контекст из kernel во время звонков. Код kernel никуда не уходит:
ни в GitHub, ни в облако. Ничего не пушь.

ПРАВИЛА
1. Новая ветка feature/companion от текущей, маленькие коммиты. Существующее поведение не менять. Каталог данных и
   конфиги с токенами не коммитить и не выгружать.
2. Токены и пароли не логировать и не включать ни в один выгружаемый текст. Перед выгрузкой прогонять тексты через
   call_companion.redact.Redactor.
3. Если не уверен в метке чувствительности источника — выбери более строгую и спроси меня.
4. СНАЧАЛА изучи репозиторий и перескажи мне структуру: где лежат сообщения чатов, заметки, данные об активности и
   фокусе, карточки и задачи, календарь, GitHub, браузер, платежи; как устроены хранилище и CLI, как работает фоновый
   воркер. Дождись моего «ок» и только потом пиши код.

ЧТО ВЗЯТЬ
Склонируй https://github.com/dansichka/start, ветка claude/smart-call-companion-4icjl5, во временную папку. Из
call-companion/src/call_companion/ скопируй в kernel/companion/vendor/call_companion/ файлы: __init__.py, models.py,
util.py, bm25.py, redact.py, kernel_api.py, kernel_server.py. Нужны pydantic>=2.7 и fastapi, uvicorn (для HTTP), Python 3.10+.
Спецификация контракта — call-companion/docs/KERNEL_CONTRACT.md в той же ветке. Прочитай её целиком.

ЧТО РЕАЛИЗОВАТЬ
(a) kernel/companion/source.py: класс KernelDataSource с методами stats, search, brief, upcoming, save_note, run_action.
    Каждую запись kernel превращай в ContextItem: стабильный id вида «источник:id», source, kind, title, text (до 1000
    символов), ts, url, entities (люди, компании, проекты), sensitivity. Особые kind: open_loop для моих невыполненных
    обещаний, talking_point для заготовленных тезисов. save_note пишет в заметки kernel с тегом from:call-companion и
    пометкой «не проверено». run_action поддерживает только capability kernel.task (создать задачу или карточку, id действия
    — ключ идемпотентности), остальное — исключение.
(b) CLI kernel: группа «companion» с командами health, search, brief, upcoming, note, action — это
    run_cli(KernelDataSource(), [op]); команда «companion serve» (HTTP через kernel_server.serve; токен из переменной
    KERNEL_COMPANION_TOKEN, без него не стартовать; адрес из конфигурации kernel, по умолчанию loopback) и «companion export
    --days 30 --out PATH» (write_pack).
(c) Периодическая выгрузка пакета раз в 30 минут из существующего воркера, путь из конфигурации kernel.
(d) Тесты по образцу call-companion/tests/test_kernel_contract.py: клиент против create_app(KernelDataSource(тест-данные),
    токен). Проверить: secret не возвращается никогда, max_sensitivity соблюдается, заметка сохраняется как не проверенная,
    неверный токен даёт 401, в выгрузке нет секретов из конфигов.

КРИТЕРИИ ГОТОВНОСТИ
Тесты проходят. «companion export --out pack.jsonl» создаёт файл, который принимает «callc pack-info pack.jsonl»
(пропущено строк 0). Покажи мне вывод тестов и pack-info, список изменённых файлов и как это запустить.
```

## Частые вопросы

**ПК выключен во время звонка.** Компаньон работает по `pack.jsonl` и досье; в HUD появится предупреждение, что живой kernel не ответил.

**Заметки, сделанные на звонке, не дошли.** Если HTTP недоступен, они лежат в `pack.jsonl.outbox.jsonl` рядом с пакетом. Это обычный JSONL, kernel может импортировать его при следующей синхронизации.

**Версионирование.** Пути начинаются с `/v1`. Несовместимые изменения пойдут под `/v2`, лишние поля клиент игнорирует.
