# Advanced CrewAI з MCP Sequential Thinking

**Продакшен-ready** multi-agent система для аналізу новин з офіційним MCP Sequential Thinking сервером.

## 🎯 Що робить цей агент?

1. **Паралельний пошук** - одночасно шукає новини з 3 джерел (BBC, CNN, Reuters)
2. **MCP Sequential Thinking** - використовує офіційний MCP сервер для структурованого аналізу
3. **Синтез висновків** - генерує комплексний звіт з рекомендаціями

## 📊 Архітектура

### 5-Агентна система з MCP:

```
┌─────────────────┐
│ BBC Researcher  │─┐
└─────────────────┘ │
                    ├──► ┌──────────────────┐       ┌─────────────────┐
┌─────────────────┐ │    │ Senior Analyst   │       │ MCP Sequential  │
│ CNN Researcher  │─┤    │ (з MCP tools)    │◄──────┤ Thinking Server │
└─────────────────┘ │    └──────────────────┘       └─────────────────┘
                    │                ▼                (npx @model...)
┌─────────────────┐ │    ┌──────────────────┐
│Reuters Researcher│─┘    │ Report Synthesizer│
└─────────────────┘       └──────────────────┘
```

### MCP Sequential Thinking:

CrewAI автоматично підключається до офіційного MCP Sequential Thinking сервера через `MCPServerAdapter`:

```python
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-sequential-thinking"]
)

with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
    analyst_agent = Agent(
        role='Senior News Analyst',
        tools=mcp_tools,  # Автоматично отримує sequentialthinking tool
        ...
    )
```

## 📋 Структура проекту

```
module4/
├── README.md                        # Документація
├── requirements.txt                 # Python залежності (crewai-tools[mcp]!)
├── .env.example                     # Шаблон конфігурації
├── .gitignore                       # Git exclusions
│
├── config.py                        # Управління налаштуваннями
├── parallel_agent_with_mcp.py      # 🎯 Основний агент (5 агентів + MCP)
└── mcp_examples.py                 # 📚 Приклади різних MCP серверів
```

## 🚀 Встановлення

### Крок 1: Node.js (для MCP сервера)

MCP Sequential Thinking сервер працює через npx. Перевірте:

```bash
node --version  # v18+ рекомендовано
npx --version
```

Якщо немає Node.js: [https://nodejs.org/](https://nodejs.org/)

### Крок 2: Python залежності

```bash
pip install -r requirements.txt
```

**Залежності:**
- `crewai>=0.80.0` - Multi-agent framework
- `crewai-tools[mcp]>=1.3.0` - MCP адаптер для CrewAI
- `langchain>=1.0.0` - LLM orchestration
- `langchain-openai>=1.0.0` - OpenAI integration
- `mcp>=1.6.0` - Model Context Protocol
- `requests>=2.31.0` - HTTP requests (для Brave Search API)
- `python-dotenv>=1.0.0` - Environment variables

### Крок 3: Конфігурація

```bash
cp .env.example .env
```

Відредагуйте `.env`:
```bash
# Обов'язково
OPENAI_API_KEY=your-openai-api-key-here
BRAVE_API_KEY=your-brave-api-key-here  # Отримайте на https://brave.com/search/api/

# Опціонально
DEFAULT_MODEL=gpt-4o-mini
TEMPERATURE=0.7
ENABLE_MCP_THINKING=true
MAX_SEARCH_RESULTS=3
```

## 🎮 Використання

### Базовий запуск

```bash
python parallel_agent_with_mcp.py
```

**Очікуваний вивід:**
```
🔌 Підготовка до підключення MCP Sequential Thinking сервера...
================================================================================
🚀 ADVANCED CREWAI: Паралельний пошук + MCP Sequential Thinking
================================================================================

📋 Створення пошукових агентів...

🔍 Тема аналізу: 'artificial intelligence breakthrough'
🧠 MCP Sequential Thinking: Підключаємось...
   └─ Пошук з BBC, CNN, Reuters (паралельно)

✅ MCP сервер підключено! Доступно інструментів: 1
   📋 Інструменти:
      • sequentialthinking

⚡ Запуск паралельного пошуку та аналізу...
```

### MCP Sequential Thinking в дії

Агент автоматично використовує MCP сервер для 5-крокового аналізу:

```
Sequential Thinking MCP Server running on stdio

┌────────────────────────────────────────────────┐
│ 💭 Thought 1/5                                 │
├────────────────────────────────────────────────┤
│ Визначаю основні теми що згадуються в усіх     │
│ джерелах: AI regulations, ethical concerns...  │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ 💭 Thought 2/5                                 │
├────────────────────────────────────────────────┤
│ Знаходжу унікальні інсайти з кожного джерела...│
└────────────────────────────────────────────────┘
...
```

### Програмне використання

```python
from parallel_agent_with_mcp import run_advanced_analysis

# Запустити аналіз
result = run_advanced_analysis(
    topic="quantum computing breakthrough"
)

print(result['result'])
print(f"Час виконання: {result['duration']:.2f}с")
print(f"MCP enabled: {result['mcp_enabled']}")
```

## 🔧 Як працює MCP інтеграція

### 1. MCPServerAdapter підключається до npx сервера

```python
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
    env=None
)

with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
    # mcp_tools містить список інструментів з MCP сервера
    analyst_agent = create_analyst_agent_with_mcp(mcp_tools)
```

### 2. Агент викликає MCP інструмент

Аналітик отримує інструкції викликати `sequentialthinking`:

```python
analysis_description = f'''
Використай інструмент "sequentialthinking" для структурованого аналізу.

Крок 1 - thought: "Визначаю основні теми..."
        thoughtNumber: 1, totalThoughts: 5, nextThoughtNeeded: true

Крок 2 - thought: "Знаходжу унікальні інсайти..."
        thoughtNumber: 2, totalThoughts: 5, nextThoughtNeeded: true
...
'''
```

### 3. MCP сервер візуалізує процес

Офіційний MCP сервер автоматично виводить красиві боксики для кожного кроку мислення в консоль.

## 💡 Переваги MCP підходу

✅ **Офіційний сервер** - використовується @modelcontextprotocol/server-sequential-thinking
✅ **Автоматична візуалізація** - MCP сервер сам малює thinking boxes
✅ **Вбудована підтримка CrewAI** - через MCPServerAdapter
✅ **Стандартизований протокол** - Model Context Protocol
✅ **Легко розширювати** - можна додати інші MCP сервери

## 🧪 Вимоги

- Python 3.10+
- Node.js 18+ (для npx)
- OpenAI API Key
- Інтернет з'єднання (для пошуку новин та npx)

## 🌟 Інші популярні MCP сервери

Цей проект демонструє Sequential Thinking, але MCP підтримує багато інших серверів!

### 📋 Офіційні MCP сервери:

#### 1. **Sequential Thinking** ⭐ (використовується в цьому проекті)
```bash
npx -y @modelcontextprotocol/server-sequential-thinking
```
Структуроване покрокове мислення для розв'язання складних проблем.

#### 2. **Filesystem**
```bash
npx -y @modelcontextprotocol/server-filesystem /path/to/allowed/dir
```
Безпечна робота з файловою системою з контролем доступу.
- Читання/запис файлів
- Створення директорій
- Пошук файлів

#### 3. **Fetch**
```bash
npx -y @modelcontextprotocol/server-fetch
```
Завантаження та конвертація веб-контенту для LLM.
- Завантаження HTML сторінок
- Конвертація в markdown
- Робота з PDF

#### 4. **Git**
```bash
npx -y @modelcontextprotocol/server-git --repository /path/to/repo
```
Робота з Git репозиторіями.
- Читання комітів
- Пошук по історії
- Аналіз змін

#### 5. **Memory**
```bash
npx -y @modelcontextprotocol/server-memory
```
Knowledge graph система пам'яті.
- Створення сутностей
- Зв'язки між концепціями
- Зберігання знань

### 🎯 Популярні сторонні MCP сервери:

- **Google Drive** - робота з Google Drive файлами
- **Slack** - інтеграція зі Slack
- **GitHub** - розширена робота з GitHub
- **PostgreSQL** - робота з базами даних
- **MongoDB** - NoSQL бази даних
- **Puppeteer** - автоматизація браузера
- **Brave Search** - пошук через Brave
- **AWS** - інтеграція з AWS сервісами
- **Azure** - Microsoft Azure інтеграція
- **Cloudflare** - Cloudflare Workers

### 💡 Приклади використання

Створено файл `mcp_examples.py` з прикладами використання різних MCP серверів:

```bash
python mcp_examples.py
```

Приклади включають:
1. **Sequential Thinking** - структуроване мислення
2. **Filesystem** - робота з файлами
3. **Fetch** - завантаження веб-контенту
4. **Git** - аналіз репозиторіїв
5. **Memory** - knowledge graph

### 🔗 Як додати інший MCP сервер

```python
from crewai import Agent
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters

# Параметри для іншого MCP сервера
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-НАЗВА"]
)

# Використання
with MCPServerAdapter(server_params, connect_timeout=60) as mcp_tools:
    agent = Agent(
        role='Your Role',
        tools=mcp_tools,  # Агент отримує інструменти з MCP
        ...
    )
```

### 📊 Повний список серверів:

- [Офіційні MCP сервери](https://github.com/modelcontextprotocol/servers)
- [MCP сервери від спільноти](https://github.com/punkpeye/awesome-mcp-servers)
- [Marketplace](https://modelcontextprotocol.io/examples)

## 📚 Корисні посилання

- [CrewAI MCP Documentation](https://docs.crewai.com/en/mcp/overview)
- [MCP Sequential Thinking Server](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [CrewAI Documentation](https://docs.crewai.com)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## 🐛 Troubleshooting

### "OpenAI API key not found"

**Проблема:** OPENAI_API_KEY не встановлений

**Рішення:**
```bash
# В .env файлі
OPENAI_API_KEY=your-api-key-here

# Або через export
export OPENAI_API_KEY="your-api-key-here"
```

### "command not found: npx"

**Проблема:** Node.js не встановлений

**Рішення:**
```bash
# macOS
brew install node

# або завантажте з https://nodejs.org/
```

### "Module 'crewai_tools' not found"

**Проблема:** crewai-tools[mcp] не встановлений

**Рішення:**
```bash
pip install 'crewai-tools[mcp]'
```

### Timeout при підключенні до MCP

**Проблема:** MCP сервер не встигає запуститися

**Рішення:**
- Перевірте що npx працює: `npx -y @modelcontextprotocol/server-sequential-thinking --version`
- Збільште timeout в коді:
  ```python
  with MCPServerAdapter(server_params, connect_timeout=120) as mcp_tools:
  ```

## 📝 License

MIT License
